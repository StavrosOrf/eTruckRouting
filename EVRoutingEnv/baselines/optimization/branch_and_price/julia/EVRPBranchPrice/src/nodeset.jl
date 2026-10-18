# Fixed-width bit set stored inline in labels, over indices 1..64W.
# Customers use 1..n and the depot n+1, so one width covers ng memories and
# succession restrictions alike.

struct NodeSet{W}
    w::NTuple{W,UInt64}
end

@inline NodeSet{W}() where {W} = NodeSet{W}(ntuple(_ -> zero(UInt64), Val(W)))

function nodeset(::Val{W}, items) where {W}
    s = NodeSet{W}()
    for i in items
        s = with(s, i)
    end
    s
end

@inline function Base.in(i::Int, s::NodeSet{W}) where {W}
    k = (i - 1) >> 6
    @inbounds ((s.w[k+1] >> ((i - 1) & 63)) & one(UInt64)) == one(UInt64)
end

@inline function with(s::NodeSet{W}, i::Int) where {W}
    k = ((i - 1) >> 6) + 1
    bit = one(UInt64) << ((i - 1) & 63)
    NodeSet{W}(ntuple(j -> j == k ? s.w[j] | bit : s.w[j], Val(W)))
end

@inline Base.:&(a::NodeSet{W}, b::NodeSet{W}) where {W} =
    NodeSet{W}(ntuple(j -> a.w[j] & b.w[j], Val(W)))

@inline function Base.issubset(a::NodeSet{W}, b::NodeSet{W}) where {W}
    acc = zero(UInt64)
    @inbounds for j in 1:W
        acc |= a.w[j] & ~b.w[j]
    end
    acc == zero(UInt64)
end

@inline Base.isempty(s::NodeSet{W}) where {W} = all(==(zero(UInt64)), s.w)

words_for(count::Int) = max(1, cld(count, 64))
