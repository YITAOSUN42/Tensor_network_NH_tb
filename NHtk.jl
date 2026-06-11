using ITensors
using ITensorMPS
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using Quantics
include("2D_lattice.jl") 

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
    k_mpo_1 = MPO(kinetic_1,sites)
    
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
    return k_mpo_1, k_mpo_2
end

#Now KPM 
#n is N/2
function KPM_Tn_NH(H,n,scale,Id_op)
    N = 2* n 
    Ham_n = H/scale

    T_k_minus_2 = Id_op
    T_k_minus_1 = Ham_n

    T_k_minus_1_partial = I_ldn 
    T_k_minus_2_partial = 0*I_rup

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
       
            

            T_k_partial = +( apply(2 * I_ldn, T_k_minus_1; ) , 2* apply(Ham_n,T_k_minus_1_partial; );  maxdim=80)
            T_k_partial = +(T_k_partial, -T_k_minus_2_partial; maxdim=80)
            
            T_k_minus_2_partial = T_k_minus_1_partial
            T_k_minus_1_partial = T_k_partial
            println(maxlinkdim(T_k_partial))

            T_k = +(2 * apply(Ham_n, T_k_minus_1;) , -T_k_minus_2; maxdim=80) 

            T_k_minus_2 = T_k_minus_1 
            T_k_minus_1 =  T_k  
            push!(Tn_paritial_list,T_k_partial)
        end
    end
    return   Tn_paritial_list 
end
 
function get_energy_from_Tnp(Tn_partial_list,nn)
  
    N = 2*nn
    jackson_kernel = [(N - n+1) * cos(π * n / (N+1)) + sin(π * n / (N+1)) / tan(π / (N+1)) for n in 0:N-1 ]
 
    # Compute electronic density
    A = Tn_partial_list[1]  
    for l in 2:2:N
        order = (-1)^(((l)/2 -1))
        A = +(A,  order *  Tn_partial_list[l]  * jackson_kernel[l-1] ; maxdim=100)
      
        
    end
    
    A  *= 2/(π^2 * (N+1))
    rotate_A = apply(I_rup,A)
    dos_point = tr(rotate_A)

    #B = 0
    #for l in 2:2:N
    #    order = (-1)^(((l)/2 -1))
    #   inte_MPO = apply(I_rup,Tn_partial_list[l])
     #   B +=  order * jackson_kernel[l-1] * tr(inte_MPO)

   # end

    #B  *= 2/(π^2 * (N+1))
    
    return rotate_A , dos_point
end

function interchain_hopping_square_ssh(L_chain,hop_inter_1, hop_inter_2, num_site, sites) 
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
    k_mpo_1_t = apply(hop_inter_1,K_mpo_1_true)

    #k_mpo_1_t = apply(k_mpo_2,k_mpo_1_t)


    #shift it
    K_mpo_2_ture =  exponential_shift(k_mpo_2, shifting_order)
    k_mpo_2_t = apply(K_mpo_2_ture,hop_inter_2)
    #k_mpo_2_t = apply(k_mpo_2_t ,k_mpo_1)
    
    k_mpo = k_mpo_1_t + k_mpo_2_t 

    return k_mpo 
end

function intrachain_hopping_nh(L_chain, hop_intra_1,hop_intra_2,num_site, sites) 

    L = Int(log2(num_site))
    break_mpo = break_chain(L_chain, num_site, sites)
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
    k_mpo_1 = MPO(kinetic_1,sites)
    k_cen = apply(hop_intra_1,k_mpo_1)
    true_hop_1 = apply(break_mpo, k_cen)

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
    k_cen = apply( k_mpo_2,hop_intra_2)
    true_hop_2 = apply(k_cen, break_mpo)
    
    k_mpo =  +(true_hop_1, true_hop_2;  cutoff = 1e-8)  

    return k_mpo 
end

function interchain_hopping_square_ssh_2(L_chain,hop_inter_1,hop_inter_2, num_site, sites) 
    #the hamiltonian for (3)
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


    k_mpo_1_diag = apply(hop_inter_1,K_mpo_1_true)
    k_mpo_2_diag = apply(hop_inter_2,K_mpo_1_true)

    k_mpo_1_t = apply(k_mpo_2,k_mpo_1_diag)
    k_mpo_1_d = apply(k_mpo_2_diag,k_mpo_1)


    #shift it
    K_mpo_2_ture =  exponential_shift(k_mpo_2, shifting_order) 
    k_mpo_3_diag = apply(K_mpo_2_ture,hop_inter_1)
    k_mpo_4_diag = apply(K_mpo_2_ture,hop_inter_2)

    k_mpo_2_t = apply(k_mpo_3_diag ,k_mpo_1)
    k_mpo_2_d = apply(k_mpo_2 ,k_mpo_4_diag)

    k_mpo = k_mpo_1_t + k_mpo_2_t  + k_mpo_1_d+ k_mpo_2_d 

    return k_mpo 
end

function hermitized_mpo(omega,tot_mpo)
    upper_block = apply( k_mpo_3,(omega*Id_op - tot_mpo));
    
    lower_block = apply( swapprime(dag(upper_block ), 0 => 1),k_mpo_3);
    hermitized_ma = apply(upper_block,I_rup) + apply(I_ldn,lower_block) ;
    return hermitized_ma
end

function get_spectrum(xlims, nx, ylims, ny, npole, tot_mpo,scale)
    xgrid = range(xlims[1], xlims[2], length=nx)
    ygrid = range(ylims[1], ylims[2], length=ny)

    Z = zeros(Complex{Float64}, ny, nx)   # 用来存 DOS，行对应 y，列对应 x

    for (ix, ex) in enumerate(xgrid)
        for (iy, ey) in enumerate(ygrid)
            omega = ex + 1im * ey
            H_mpo = hermitized_mpo(omega, tot_mpo)         # 注意避免函数名和变量名重名
            p_list = KPM_Tn_NH(H_mpo, npole,scale)
            dos_point = get_energy_from_Tnp(p_list, npole)

            Z[iy, ix] = dos_point
        end
    end

    return xgrid, ygrid, Z
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
