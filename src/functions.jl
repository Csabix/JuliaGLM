# Function list (and some implementations) were collected from Chapter 8 of
# https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf

const GenFType = Union{Float32,VecTN{Float32}}
const GenDType = Union{Float64,VecTN{Float64}}
const GenIType = Union{Int32,  VecTN{Int32}}
const GenUType = Union{UInt32, VecTN{UInt32}}
const GenBType = Union{Bool,   VecTN{Bool}}

# Angle and Trigonometry Functions (8.1)

radians(degrees::GenFType) = pi .* degrees ./ 180
degrees(radians::GenFType) = 180 .* radians ./ pi

sin(angle::GenFType)                 = Base.sin.(angle)
cos(angle::GenFType)                 = Base.cos.(angle)
tan(angle::GenFType)                 = Base.tan.(angle)
atan(y::T, x::T) where {T<:GenFType} = Base.atan.(y, x)
atan(y_over_x::GenFType)             = Base.atan.(y_over_x)
asin(x::GenFType)                    = Base.asin.(x)
acos(x::GenFType)                    = Base.acos.(x)
sinh(x::GenFType)                    = Base.sinh.(x)
cosh(x::GenFType)                    = Base.cosh.(x)
tanh(x::GenFType)                    = Base.tanh.(x)
asinh(x::GenFType)                   = Base.asinh.(x)
acosh(x::GenFType)                   = Base.acosh.(x)
atanh(x::GenFType)                   = Base.atanh.(x)

# Exponential Functions (8.2)

pow(x::T, y::T) where {T<:GenFType}      = x .^ y
exp(x::GenFType)                         = Base.exp.(x)
exp2(x::GenFType)                        = Base.exp2.(x)
log(x::GenFType)                         = Base.log.(x)
log2(x::GenFType)                        = Base.log2.(x)
sqrt(x::Union{GenFType,GenDType})        = Base.sqrt.(x)
inversesqrt(x::Union{GenFType,GenDType}) = Base.inv.(sqrt(x))

# Common Functions (8.3)

abs(x::Union{GenFType,GenIType,GenDType})                    = Base.abs.(x)
sign(x::Union{GenFType,GenIType,GenDType})                   = Base.sign.(x)
floor(x::Union{GenFType,GenDType})                           = Base.floor.(x)
trunc(x::Union{GenFType,GenDType})                           = Base.trunc.(x)
roundEven(x::Union{GenFType,GenDType})                       = Base.round.(x, RoundNearest)
round(x::Union{GenFType,GenDType})                           = roundEven(x) # round ≡ roundEven is allowed by the spec
ceil(x::Union{GenFType,GenDType})                            = Base.ceil.(x)
fract(x::Union{GenFType,GenDType})                           = x .- floor(x)
mod(x::T, y::T)          where {T<:Union{GenFType,GenDType}} = Base.mod.(x, y)
mod(x::VecNT{N,T}, y::T) where {N,T<:Union{Float32,Float64}} = mod(x, similar_type(x)(y))

function modf(x::T, i::Ref{T}) where {T<:Union{Float32,Float64}}
    (frac, int) = Base.modf(x)

    i[] = int
    frac
end

function modf(x::VecNT{N,T}, i::Ref{<:VecNT{N,T}}) where {N,T<:Union{Float32,Float64}}
    ints = MVector{N,T}(undef)
    fracs = MVector{N,T}(undef)

    for idx in 1:N
        (frac, int) = Base.modf(x[idx])

        fracs[idx] = frac
        ints[idx] = int
    end

    i[] = similar_type(x)(ints...)
    similar_type(x)(fracs...)
end

min(x::T, y::T)          where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = Base.min.(x, y)
min(x::VecNT{N,T}, y::T) where {N,T<:Union{Float32,Float64,Int32,UInt32}}      = min(x, similar_type(x)(y))
max(x::T, y::T)          where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = Base.max.(x, y)
max(x::VecNT{N,T}, y::T) where {N,T<:Union{Float32,Float64,Int32,UInt32}}      = max(x, similar_type(x)(y))

clamp(x::T, min_val::T, max_val::T) where {T<:Union{GenFType,GenDType,GenIType,GenUType}} =
    min(max(x, min_val), max_val)
clamp(x::VecNT{N,T}, min_val::T, max_val::T) where {N,T<:Union{Float32,Float64,Int32,UInt32}} =
    clamp(x, similar_type(x)(min_val), similar_type(x)(max_val))

mix(x::T, y::T, a::T) where {T<:Union{GenFType,GenDType}} =
    x .* (one(eltype(T)) .- a) + y .* a
mix(x::VecNT{N,T}, y::VecNT{N,T}, a::T) where {N,T<:Union{Float32,Float64}} =
    mix(x, y, similar_type(x)(a))
mix(x::VecNT{N,T}, y::VecNT{N,T}, a::VecNT{N,Bool}) where {N,T<:StaticNumber} =
    similar_type(x)([selector ? y_el : x_el for (selector, x_el, y_el) in zip(a, x, y)]...)

step(edge::VecNT{N,T}, x::VecNT{N,T}) where {N,T<:Union{Float32,Float64}} =
    similar_type(x)([x_el < edge_el ? zero(T) : one(T) for (edge_el, x_el) in zip(edge, x)]...)
step(edge::T, x::VecNT{N,T}) where {N,T<:Union{Float32,Float64}} =
    step(similar_type(x)(edge), x)

function smoothstep(edge0::T, edge1::T, x::T) where {T<:Union{GenFType,GenDType}}
    t = clamp((x - edge0) ./ (edge1 - edge0), zero(eltype(T)), one(eltype(T)))
    t .^ 2 .* (3 .- 2 .* t)
end

smoothstep(edge0::T, edge1::T, x::VecNT{N,T}) where {N,T<:Union{Float32,Float64}} =
    smoothstep(similar_type(x)(edge0), similar_type(x)(edge1), x)

# Geometric Functions (8.5)

length(x::Union{GenFType,GenDType})                                     = sum(x .^ 2) |> sqrt
normalize(x::Union{GenFType,GenDType})                                  = LinearAlgebra.normalize(x)
distance(p0::T, p1::T)              where {T<:Union{GenFType,GenDType}} = length(p0 .- p1)
dot(x::T, y::T)                     where {T<:Union{GenFType,GenDType}} = LinearAlgebra.dot(x, y)
cross(x::VecNT{3,T}, y::VecNT{3,T}) where {T<:Union{Float32,Float64}}   = LinearAlgebra.cross(x, y)
faceforward(N::T, I::T, Nref::T)    where {T<:Union{GenFType,GenDType}} = dot(Nref, I) < 0 ? N : -N

function reflect(I::T, N::T) where {T<:Union{GenFType,GenDType}}
    @assert length(N) ≈ 1 "reflect must be called with a normalized surface normal, it was called with vec of length $(length(N))"
    I .- 2 .* dot(N, I) .* N
end

function refract(I::VecNT{VN,T}, N::VecNT{VN,T}, eta::T) where {VN,T<:Union{Float32,Float64}}
    @assert length(N) ≈ 1 "refract must be called with a normalized surface normal, it was called with vec of length $(length(N))"
    @assert length(I) ≈ 1 "refract must be called with a normalized incident vector, it was called with vec of length $(length(I))"

    k = one(T) - eta * eta * (one(T) - dot(N, I) * dot(N, I))
    if k < zero(T)
        return similar_type(I)(zero(T))
    end

    eta .* I .- (eta * dot(N, I) + sqrt(k)) * N
end

# Vector Relational Functions (8.7)

lessThan(x::T, y::T)         where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = x .< y
lessThanEqual(x::T, y::T)    where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = x .<= y
greaterThan(x::T, y::T)      where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = x .> y
greaterThanEqual(x::T, y::T) where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = x .>= y
equal(x::T, y::T)            where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = x .== y
notEqual(x::T, y::T)         where {T<:Union{GenFType,GenDType,GenIType,GenUType}} = x .!= y

any(x::GenBType) = Base.any(x)
all(x::GenBType) = Base.all(x)
not(x::GenBType) = map(el -> !el, x)