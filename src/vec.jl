
#
#   Vec N
#

const VecNT{N,T<:StaticNumber} = SVector{N, T}
const VecTN{T,N} = VecNT{N,T}

const Vec2T{T} = VecNT{2,T}
const Vec3T{T} = VecNT{3,T}
const Vec4T{T} = VecNT{4,T}

StaticArrays.similar_type(::Type{<:Vec2T}, ::Type{T}, s::Size{(2,)}) where T = Vec2T{T}
StaticArrays.similar_type(::Type{<:Vec3T}, ::Type{T}, s::Size{(3,)}) where T = Vec3T{T}
StaticArrays.similar_type(::Type{<:Vec4T}, ::Type{T}, s::Size{(4,)}) where T = Vec4T{T}

Vec2T{T}(x::StaticNumber) where T = Vec2T{T}(x,x)
Vec3T{T}(x::StaticNumber) where T = Vec3T{T}(x,x,x)
Vec4T{T}(x::StaticNumber) where T = Vec4T{T}(x,x,x,x)
Vec3T{T}(x::StaticNumber,yz::Vec2T) where T = Vec3T{T}(x,yz...)
Vec3T{T}(xy::Vec2T,z::StaticNumber) where T = Vec3T{T}(xy...,z)
Vec4T{T}(x::StaticNumber,y::StaticNumber,zw::Vec2T) where T = Vec4T{T}(x,y,zw...)
Vec4T{T}(x::StaticNumber,yz::Vec2T,w::StaticNumber) where T = Vec4T{T}(x,yz...,w)
Vec4T{T}(xy::Vec2T,z::StaticNumber,w::StaticNumber) where T = Vec4T{T}(xy...,z,w)
Vec4T{T}(x::StaticNumber,yzw::Vec3T) where T = Vec4T{T}(x,yzw...)
Vec4T{T}(xyz::Vec3T,w::StaticNumber) where T = Vec4T{T}(xyz...,w)
Vec4T{T}(xy::Vec2T,zw::Vec2T) where T = Vec4T{T}(xy...,zw...)
# Actually glsl has oversized vec constructors, such as vec2(vec3(1,2,3))=vec2(1,2). I don't wanna support that.

export Vec2T, Vec3T, Vec4T, VecTN, VecNT

# Generate concrete types and constructors
for n in 2:4
    dsym = Symbol("Vec"*string(n)*"T")
    for (str, type) in CharTypeMap
        ssym = Symbol(uppercase(str)*"Vec"*string(n))
        fsym = Symbol(lowercase(str)*"vec"*string(n))
        @eval const $ssym = $dsym{$type}
        @eval @inline $fsym(v...) = $ssym(v...) 
        @eval export $ssym, $fsym
    end
end

# SWIZZLE
import StaticArrays.getindex

@Base.propagate_inbounds @inline function getindex(v::VecNT{N,T},ii::NTuple{1}) where {N,T}
    v[@inbounds ii[1]]
end
@Base.propagate_inbounds @inline function getindex(v::VecNT{N,T},ii::NTuple{2}) where {N,T}
    Vec2T{T}(v[@inbounds ii[1]],v[@inbounds ii[2]])
end
@Base.propagate_inbounds @inline function getindex(v::VecNT{N,T},ii::NTuple{3}) where {N,T}
    Vec3T{T}(v[@inbounds ii[1]],v[@inbounds ii[2]],v[@inbounds ii[3]])
end
@Base.propagate_inbounds @inline function getindex(v::VecNT{N,T},ii::NTuple{4}) where {N,T}
    Vec4T{T}(v[@inbounds ii[1]],v[@inbounds ii[2]],v[@inbounds ii[3]],@inbounds v[ii[4]])
end

macro def_swizzle(type, max_len, chars_tuple)
    chars = [eval(arg) for arg in chars_tuple.args]
    ast = :(getfield(v, sym))
    
    for len in max_len:-1:1
        for c in Iterators.product(ntuple(_ -> chars, len)...)
            sym_name = Symbol(join(c))
            char_map = (4,1,2,3)
            indices = [char_map[Int(ch) - 118] for ch in c]
            L = Base.length(indices)
            
            if L == 1
                val_expr = :(@inbounds v[$(indices[1])])
            else
                target_type = Symbol(:Vec, L, :T)
                args = [:(@inbounds v[$i]) for i in indices]
                val_expr = Expr(:call, target_type, args...)
            end
            
            ast = :(sym === $(QuoteNode(sym_name)) ? $val_expr : $ast)
        end
    end
    
    return quote
        @inline function Base.getproperty(v::$type, sym::Symbol)
            return $ast
        end
    end
end

@def_swizzle Vec2T 4 ('x', 'y')
@def_swizzle Vec3T 4 ('x', 'y', 'z')
@def_swizzle Vec4T 4 ('x', 'y', 'z', 'w')

export getindex

const Vec4F = Vec4T{Float32}
const Vec3F = Vec3T{Float32}
const Vec2F = Vec2T{Float32}

export Vec2F, Vec3F, Vec4F

const Vec4D = Vec4T{Float64}
const Vec3D = Vec3T{Float64}
const Vec2D = Vec2T{Float64}

export Vec2D, Vec3D, Vec4D
