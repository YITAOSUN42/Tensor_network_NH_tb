using ITensors
using ITensorMPS
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using Quantics
include("2D_lattice.jl") 

function get_energy_from_T_MPS(Tn_partial_list,nn,index,sites)
  
    N = 2*nn
    jackson_kernel = [(N - n+1) * cos(π * n / (N+1)) + sin(π * n / (N+1)) / tan(π / (N+1)) for n in 0:N-1 ]
 
    # Compute electronic density
    A = Tn_partial_list[1]  
    for l in 2:2:N
        order = (-1)^(((l)/2 -1))
        A = +(A,  order *  Tn_partial_list[l]  * jackson_kernel[l-1] ; maxdim=100)
      
        
    end
    
    A  *= 2/(π^2 * (N+1))
    up_vec,dn_vec = up_down_vecs(index,sites) 
    dos_point = inner(dn_vec',A)
    
    return  dos_point
end

function shifting_ops(L,sites)
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end

        kinetic_1 += os
    end
    k_mpo_1 = MPO(ComplexF64,kinetic_1,sites)
    
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end

        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end

        kinetic_2 += os
    end

    k_mpo_2 = MPO(ComplexF64,kinetic_2,sites)
    return k_mpo_1, k_mpo_2
end

#Now KPM 
#n is N/2
function KPM_Tn_NH(H,n,scale,maxdims)
    N = 2* n 
    Ham_n =  H /scale

    T_k_minus_2 =  Id_op 
    T_k_minus_1 = Ham_n

    T_k_minus_1_partial =   I_ldn 
    T_k_minus_2_partial = 0* (I_rup)

    Tn_list = [T_k_minus_2,T_k_minus_1]
    Tn_paritial_list = [T_k_minus_2_partial,T_k_minus_1_partial]

    for k in 1:N
        if k == 1
            T_k = T_k_minus_2
            T_k_partial = T_k_minus_2_partial 
        elseif k == 2
            T_k = T_k_minus_1
            T_k_partial = T_k_minus_1_partial 
        else
       
            

            T_k_partial = add( apply(2 * (I_ldn), T_k_minus_1;maxdim = maxdims) , 2* apply(Ham_n,T_k_minus_1_partial;maxdim = maxdims); maxdim = maxdims , cutoff=1e-8) 
            #T_k_partial = ITensorMPS.truncate!(T_k_partial; maxdim = maxdims)
            T_k_partial = add(T_k_partial,-T_k_minus_2_partial;  maxdim = maxdims , cutoff=1e-8)
            #T_k_partial = ITensorMPS.truncate!(T_k_partial;maxdim = maxdims)
            T_k_minus_2_partial = T_k_minus_1_partial
            T_k_minus_1_partial = T_k_partial
         

            T_k = add(2 * apply(Ham_n, T_k_minus_1 ; maxdim = maxdims) , -T_k_minus_2;maxdim = maxdims , cutoff=1e-8 ) 
            #T_k = ITensorMPS.truncate!(T_k; maxdim = maxdims)
            T_k_minus_2 = T_k_minus_1 
            T_k_minus_1 =  T_k  
 
            push!(Tn_paritial_list,T_k_partial)
        end
    end
    return   Tn_paritial_list 
end


 
function contract_mpo_block(W::MPO, row=2, col=1)
    N = length(W) - 1
    if N < 2
        error("The MPO length must be at least 2 for this contraction.")
    end

    # 1. Get the tensor and site index for the first site.
    first_tensor = W[1]
    s = siteind(W, 1)

    # 2. Extract the requested physical block.
    # Use basis-state tensors built with state(s, i) for projection.
    # After contraction, target_block usually only keeps the Link index to site 2.
    target_block = first_tensor * dag( (state(s, col))) *  (state(s', row))

    # 3. Contract the projected block into the second tensor.
    # The left Link index of new_first_site is removed at this point.
    new_first_site = W[2] * target_block

    # 4. Build the tensor list for the reduced MPO.
    # The first site of the new MPO is the contracted old second site.
     
    W_new_list = [new_first_site]
    
    for i in 3:(N+1)

        push!(W_new_list, W[i])

    end

    # 5. Return the reduced MPO object.
    return MPO(W_new_list)
end

function extract_diagonal_to_mps(M::MPO)::MPS
    N = length(M)
    new_tensors = Vector{ITensor}(undef, N)

    for i in 1:N
        t = M[i]

        # Get the physical indices for this site (bra, ket)
        si_pair = siteinds(M, i)
        s2 = si_pair[1]   # bra index
        s1 = si_pair[2]   # ket index

        # Dimension of the physical index
        dim_s = dim(s1)

        # Collect all virtual (link) indices, leaving out the physical ones
        v_inds = uniqueinds(t, s1, s2)

        # Create a new MPS tensor with the same virtual indices and one physical index
        res = ITensor(v_inds..., s1)

        # Loop over physical states to extract diagonal elements
        for v in 1:dim_s
            # Take the slice corresponding to s1 = s2 = v
            slice = t *  (onehot(s1 => v)) *  (onehot(s2 => v))

            # Add this slice back into the resulting MPS tensor at position v
            res += slice *  (onehot(s1 => v))
        end

        # Store the resulting MPS tensor
        new_tensors[i] = res
    end

    # Build the MPS from the list of site tensors
    return MPS(new_tensors)
end

function pre_process(Tn_partial_list)

    #get the correct block
    reduced_block_lis = contract_mpo_block.(Tn_partial_list)
    #get all the MPS
    diag_mps_lis = extract_diagonal_to_mps.(reduced_block_lis)
    return diag_mps_lis
end

function ones_mps(sites::Vector{<:Index})
    N = length(sites)
    # Initialize with placeholder product-state labels; the tensor entries are
    # overwritten below.
    psi = MPS(sites, "1") 
    
    for n in 1:N
        d = dim(sites[n])
        # Create an all-ones tensor over the site's existing physical and Link
        # indices.
        ones_tensor = ITensor(1.0, inds(psi[n]))
        psi[n] = ones_tensor
    end
    
    return psi
end

function get_energy_from_Tnp(Tn_partial_list,nn,maxdims)
  
    N = 2*nn
    jackson_kernel = [(N - n+1) * cos(π * n / (N+1)) + sin(π * n / (N+1)) / tan(π / (N+1)) for n in 0:N-1 ]
    list = pre_process(Tn_partial_list)
    # Compute electronic density
    A = list[1]  
    for l in 2:2:N
        order = (-1)^(((l)/2 -1))
        A = +(A,  order *  list[l]  * jackson_kernel[l-1] ;maxdim = maxdims)
    end
    
    A  *= 2/(π^2 * (N+1))
    
   
   
    dos_point = inner( (ones_mps(siteinds(A)))',A)

 
    
    return A , dos_point 
end

function interchain_hopping_square_ssh(L_chain,hop_inter, num_site, sites) 
    L = Int(log2(num_site))
     
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end

        for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
    k_mpo_1 = MPO(kinetic_1,sites)
    #shift it
    shifting_order = Int(log2(L_chain))

    K_mpo_1_true = exponential_shift(k_mpo_1, shifting_order) 
    k_mpo_1_t = apply(hop_inter,K_mpo_1_true)

    k_mpo_1_t = apply(k_mpo_2,k_mpo_1_t)


    #shift it
    K_mpo_2_ture =  exponential_shift(k_mpo_2, shifting_order) 
    k_mpo_2_t = apply(K_mpo_2_ture,hop_inter)
    k_mpo_2_t = apply(k_mpo_2_t ,k_mpo_1)
    
    k_mpo = k_mpo_1_t + k_mpo_2_t 

    return k_mpo 
end

function hermitized_mpo(omega,tot_mpo)
    omegac = ComplexF64(omega)
    upper_block = apply( k_mpo_3,(omegac*Id_op - tot_mpo));
    lower_block = apply( dag(upper_block),k_mpo_3);
    hermitized_ma = apply(upper_block,I_rup) + apply(I_ldn,lower_block) ;
    return hermitized_ma
end

function get_spectrum(xlims, nx, ylims, ny, npole, tot_mpo,scale)
    xgrid = range(xlims[1], xlims[2], length=nx)
    ygrid = range(ylims[1], ylims[2], length=ny)

    Z = zeros(Complex{Float64}, ny, nx)   # DOS grid: rows are y, columns are x.

    for (ix, ex) in enumerate(xgrid)
        for (iy, ey) in enumerate(ygrid)
            omega = ex + 1im * ey
            H_mpo = hermitized_mpo(omega, tot_mpo)
            p_list = KPM_Tn_NH(H_mpo, npole,scale)
            dos_point = get_energy_from_Tnp(p_list, npole)

            Z[iy, ix] = dos_point
        end
    end

    return xgrid, ygrid, Z
end

function up_down_vecs(index,sites)
    N = length(sites)

    up_vec = randomMPS(ComplexF64,sites,to_binary_vector(Int(index),Int(log2(2^N))))
    dn_vec = randomMPS(ComplexF64,sites,to_binary_vector(Int(index + 2^(N-1)),Int(log2(2^N))))
    
    return up_vec, dn_vec
end
function KPM_Tn_NH_bysite(H,n,index,scale,sites,I_ldn)
    N = 2* n 
    Ham_n =  (H)/scale
    up_vec,dn_vec = up_down_vecs(index,sites)
    T_k_minus_2 =  (up_vec)
    T_k_minus_1 = apply(Ham_n,T_k_minus_2)

    T_k_minus_1_partial =  (dn_vec)
    T_k_minus_2_partial = 0* (up_vec)

    Tn_list = [T_k_minus_2,T_k_minus_1]
    Tn_paritial_list = [T_k_minus_2_partial,T_k_minus_1_partial]

    for k in 1:N
        if k == 1
            T_k = T_k_minus_2
            T_k_partial = T_k_minus_2_partial 
        elseif k == 2
            T_k = T_k_minus_1
            T_k_partial = T_k_minus_1_partial 
        else
            T_k_partial = +( apply(2 *  (I_ldn), T_k_minus_1; maxdim=100) , 2* apply(Ham_n,T_k_minus_1_partial; maxdim=100);  maxdim=100 , cutoff=1e-8)
            T_k_partial = +(T_k_partial, -T_k_minus_2_partial; maxdim=100 , cutoff=1e-8)
            
            T_k_minus_2_partial = T_k_minus_1_partial
            T_k_minus_1_partial = T_k_partial
     
            T_k = +(2 * apply(Ham_n, T_k_minus_1; maxdim=100) , -T_k_minus_2; maxdim=100 , cutoff=1e-8)

            T_k_minus_2 = T_k_minus_1 
            T_k_minus_1 =  T_k  
            push!(Tn_paritial_list,T_k_partial)
        end
    end
    return   Tn_paritial_list 
end



function get_matrix_NH(mpo,size )
    mat = zeros(ComplexF64,size, size) 
    for i in 0:size-1
        for j in 0:size-1
            element = inner(randomMPS(sites,to_binary_vector(Int(i),Int(log2(size))))',mpo,randomMPS(sites,to_binary_vector(Int(j),Int(log2(size)))))
            mat[i+1,j+1] = element
        end
    end
    return mat
end

function get_spectrum_singele(ex,ey,npole, tot_mpo,scale,maxdims)
 

    Z = zeros(Complex{Float64}, 1,1)   # Single-point DOS container.

 
    omega = ex + 1im * ey
    H_mpo = hermitized_mpo(omega, tot_mpo)
    @time p_list = KPM_Tn_NH(H_mpo, npole,scale,maxdims)
    A,dos = get_energy_from_Tnp(p_list, npole,maxdims)

    

    return  A,dos 
end
 
function get_ldos(mpo,size,tot_size )
    mat = zeros(ComplexF64,size ) 
    for i in 0:2^Int((L-1) ) -1
  
        element = inner(randomMPS(sites,to_binary_vector(Int(i),Int(log2(tot_size ))))',mpo,randomMPS(sites,to_binary_vector(Int(i),Int(log2(tot_size )))))
        mat[i+1 ] = element

    end
    return mat
end
