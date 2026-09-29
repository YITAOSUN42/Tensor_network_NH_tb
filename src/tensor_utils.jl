"""Concatenate two MPO chains as a tensor product with a dimension-one link."""
function concatenate_MPOs(
    mpo1::MPO,
    sites1::Vector{<:Index},
    mpo2::MPO,
    sites2::Vector{<:Index},
)
    n1 = length(mpo1)
    n2 = length(mpo2)
    tensors = Vector{ITensor}(undef, n1 + n2)

    for index in 1:n1
        tensors[index] = mpo1[index]
    end

    right_link = linkind(mpo1, n1)
    left_link = linkind(mpo2, 0)
    middle_link = Index(1, "Link,l=$n1")
    tensors[n1] = if isnothing(right_link)
        tensors[n1] * setelt(middle_link => 1)
    else
        replaceind(tensors[n1], right_link => middle_link)
    end

    for index in 1:n2
        tensor = mpo2[index]
        if index == 1
            tensors[n1 + index] = if isnothing(left_link)
                tensor * setelt(middle_link => 1)
            else
                replaceind(tensor, left_link => middle_link)
            end
        else
            tensors[n1 + index] = tensor
        end
    end
    return MPO(tensors), vcat(sites1, sites2)
end

"""Convert a QTT/MPS scalar field to a diagonal MPO on the same sites."""
function mps_to_diagonal_mpo(mps, sites)
    nsites = length(mps)
    nsites > 1 || throw(ArgumentError("at least two MPS sites are required"))
    mpo_tensors = Vector{ITensor}(undef, nsites)
    for index in 1:nsites
        tensor = mps[index]
        old_site = if index == 1
            uniqueind(tensor, mps[index + 1])
        elseif index == nsites
            uniqueind(tensor, mps[index - 1])
        else
            uniqueind(tensor, mps[index - 1], mps[index + 1])
        end
        site = sites[index]
        temporary_site = Index(dim(site), "temp")
        mpo_tensors[index] = (
            replaceind(tensor, old_site => temporary_site) *
            delta(temporary_site, site, site')
        )
    end
    return MPO(mpo_tensors)
end
