using JuliaGLM
using Test
using StaticArrays

@testset "Vec types        " begin
    @test VecTN{Float32,2} == VecNT{2,Float32}
    @test sizeof(Vec2T{Bool}) == sizeof(BVec2)
    @test sizeof(Vec3) == 3*sizeof(Float32) # 12
    @test sizeof(DVec2) == 2*sizeof(Float64)
    @test sizeof(Vec3) == sizeof(Vec3(1,2,3))
    @test Vec3 == typeof(vec3(1,2,3))
    @test Vec2(1,2) isa FieldVector
    @test Vec4T{Float32}(1,2,3,4) isa VecNT
end

@testset "Vec constructors " begin
    @test Vec2(1,2) == vec2(1,2)
    @test Vec3(-3,-3,-3) == Vec3(-3)
    @test zero(IVec4) == ivec4(0)
    @test dvec3(-1,0,1) == vec3(ivec2(-1,0),1)
    @test vec3(0,1,2) == vec3(0,vec2(1,2))
    @test vec4(vec3(1),0) == Vec4(1,vec3(1,1,0))
    @test vec4(vec2(3,-4),vec2(-4,-1)) == vec4(3,-vec2(4),-1)
    @test_throws DimensionMismatch dvec2(1,2,3)
    @test_throws DimensionMismatch uvec4(1,2,3)
    @test_throws DimensionMismatch vec4(vec3(),vec2())
end

@testset "Vec member access" begin
    a = rand(Vec3)
    @test a.x === a[1] && a.y === a[2] && a.z === a[3]
    @test a["y"] === a.y
    @test a[(3,)] === a.z
    @test a["yx"] == a[(2,1)]
    @test a["zx"].y == a.x
    @test a["zzzz"] == vec4(a.z)
    b = rand(IVec4)
    @test b["wzyx"]["wzyx"] == b
    @test_throws BoundsError a[4]
    @test_throws BoundsError b[(5,)]
    @test_throws BoundsError a["w"]
    @test_throws BoundsError a["u"]
    @test_throws BoundsError b["{"] # 'z'+1
    @test_throws BoundsError b["xyzwx"]
end

@testset "Mat type creation" begin
    @test MatTN{Float32,2} == MatNT{2,Float32}
    @test DMat3 == DMat3x3
    @test sizeof(IMat4) == 4*sizeof(IVec4) == 4*4*sizeof(Int32)
    @test sizeof(Mat2) == sizeof(Mat2(1,2,3,4))
    @test Mat2(1,2,3,4) isa SMatrix
    @test Mat2T{Int128}(1,2,3,4) isa Mat2T
    @test Mat3(0,0,0,0,0,0,0,0,0) == mat3(0) == zero(Mat3)
    @test Mat2(1,0,0,1) == mat2(1) == one(Mat2)
end

@testset "Others           " begin
    A = mat2(1,2,3,4)
    @test JuliaGLM.vec2(A[:,1]) == vec2(1,2)
    eye,at,up = rand(Vec3),rand(Vec3),rand(Vec3)
    @test lookat(eye,at,up)*vec4(eye,1) ≈ vec4(0,0,0,1)
    @test string(A) isa String
    @test string(eye) isa String
end

const _FN_TEST_ATOL = 1e-6

@testset "Trig Fns         " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test JuliaGLM.radians(0.0f0) == 0
    @test JuliaGLM.radians(180.0f0) ≈ pi
    @test JuliaGLM.radians(360.0f0) ≈ 2*pi
    @test JuliaGLM.degrees(0.0f0) == 0
    @test JuliaGLM.degrees(pi |> Float32) ≈ 180
    @test JuliaGLM.degrees(2 * pi |> Float32) ≈ 360

    @test JuliaGLM.sin(vec3(0, pi/2, 3*pi/2)) ≈ vec3(0, 1, -1)
    @test JuliaGLM.asin(vec3(0, 1, -1)) ≈ vec3(0, pi/2, -pi/2)
    @test JuliaGLM.cos(vec3(0, pi/2, pi)) ≈ vec3(1, 0, -1)
    @test JuliaGLM.acos(vec3(1, 0, -1)) ≈ vec3(0, pi/2, pi)
    @test JuliaGLM.tan(vec3(0, pi/4, pi/6)) ≈ vec3(0, 1, √3/3)
    @test JuliaGLM.atan(vec3(0, 1, √3/3)) ≈ vec3(0, pi/4, pi/6)
    @test JuliaGLM.atan(vec3(0, 1, √3), vec3(1, 1, 3)) ≈ vec3(0, pi/4, pi/6)

    @test JuliaGLM.sinh(vec3(0, log(2), log(3))) ≈ vec3(0, 3/4, 4/3)
    @test JuliaGLM.asinh(vec3(0, 3/4, 4/3)) ≈ vec3(0, log(2), log(3))
    @test JuliaGLM.cosh(vec3(0, log(2), log(3))) ≈ vec3(1, 5/4, 5/3)
    @test JuliaGLM.acosh(vec3(1, 5/4, 5/3)) ≈ vec3(0, log(2), log(3))
    @test JuliaGLM.tanh(vec3(0, log(2), log(3))) ≈ vec3(0, 3/5, 4/5)
    @test JuliaGLM.atanh(vec3(0, 3/5, 4/5)) ≈ vec3(0, log(2), log(3))
end

@testset "Exponential Fns  " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test JuliaGLM.pow(vec4(2), vec4(0,1,2,3)) == vec4(1,2,4,8)
    @test JuliaGLM.exp(vec3(0,1,2)) ≈ vec3(1,ℯ,ℯ^2)
    @test JuliaGLM.exp2(vec4(0,1,2,3)) ≈ vec4(1,2,4,8)
    @test JuliaGLM.log(vec3(1,ℯ,ℯ^2)) ≈ vec3(0,1,2)
    @test JuliaGLM.log2(vec4(1,2,4,8)) ≈ vec4(0,1,2,3)
    @test JuliaGLM.sqrt(vec4(1,4,9,2)) ≈ vec4(1,2,3,√2)
    @test JuliaGLM.inversesqrt(vec4(1,4,9,2)) ≈ vec4(1,0.5,1/3,1/√2)
end

@testset "Common Fns       " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test JuliaGLM.abs(vec3(-3,0,2)) == vec3(3,0,2)
    @test JuliaGLM.sign(vec3(3,0,-4)) == vec3(1,0,-1)
    @test JuliaGLM.floor(vec4(1.1,1,1.9,-1.3)) == vec4(1,1,1,-2)
    @test JuliaGLM.trunc(vec4(1.1,1,1.9,-1.3)) == vec4(1,1,1,-1)
    @test JuliaGLM.roundEven(vec4(2.4,1.6,1.5,2.5)) == vec4(2)
    @test JuliaGLM.round(vec2(2.4, 1.6)) == vec2(2)
    @test JuliaGLM.ceil(vec4(0.2,1,-1.2,-1.8)) == vec4(1,1,-1,-1)
    @test JuliaGLM.fract(vec3(1.2,2,3.8)) ≈ vec3(0.2,0,0.8)
    @test JuliaGLM.mod(vec4(1,2,3,7), vec4(2,2,2,4)) == vec4(1,0,1,3)

    int_out = Ref(vec3(0))
    frac_out = JuliaGLM.modf(vec3(1.2, 5, 3.9), int_out)
    @test int_out[] == vec3(1,5,3)
    @test frac_out ≈ vec3(0.2, 0, 0.9)

    @test JuliaGLM.min(vec3(1,2,3), vec3(3,2,1)) == vec3(1,2,1)
    @test JuliaGLM.min(vec3(1,2,3), 2.0f0) == vec3(1,2,2)
    @test JuliaGLM.max(vec3(1,2,3), vec3(3,2,1)) == vec3(3,2,3)
    @test JuliaGLM.max(vec3(1,2,3), 2.0f0) == vec3(2,2,3)
    @test JuliaGLM.clamp(vec4(3,1,3,4), vec4(1,2,1,5), vec4(5,3,2,5)) == vec4(3,2,2,5)
    @test JuliaGLM.clamp(vec3(0,2,5), 1.0f0, 3.0f0) == vec3(1,2,3)
    @test JuliaGLM.mix(vec3(0), vec3(2), vec3(0,0.5,1)) == vec3(0,1,2)
    @test JuliaGLM.mix(vec3(0), vec3(2), 0.5f0) == vec3(1)
    @test JuliaGLM.mix(vec3(0), vec3(2,1,3), bvec3(true, false, true)) == vec3(2,0,3)
    @test JuliaGLM.step(vec3(1,0,2), vec3(0,1,2)) == vec3(0,1,1)
    @test JuliaGLM.step(1.0f0, vec3(0,1,2)) == vec3(0,1,1)
    @test JuliaGLM.smoothstep(vec3(0), vec3(1,2,2), vec3(-1,3,1)) == vec3(0,1,0.5)
    @test JuliaGLM.smoothstep(0.0f0, 2.0f0, vec3(-1,3,1)) == vec3(0,1,0.5)
end

@testset "Geometry Fns     " begin
    ≈(val,expected) = isapprox(val, expected, atol=_FN_TEST_ATOL)

    @test JuliaGLM.length(vec2(1)) ≈ √2
    @test JuliaGLM.length(vec3(1)) ≈ √3
    @test JuliaGLM.normalize(vec2(1)) ≈ vec2(1/√2)
    @test JuliaGLM.normalize(vec3(1)) ≈ vec3(1/√3)
    @test JuliaGLM.distance(vec3(1), vec3(2)) ≈ √3
    @test JuliaGLM.distance(vec3(4), vec3(2)) ≈ 2√3
    @test JuliaGLM.dot(vec3(1,0,0), vec3(0,1,0)) ≈ 0
    @test JuliaGLM.dot(vec3(0.5,2,1), vec3(4,2,3)) ≈ 9
    @test JuliaGLM.cross(vec3(1,0,0), vec3(0,1,0)) ≈ vec3(0,0,1)
    @test JuliaGLM.cross(vec3(1,2,3), vec3(2,4,6)) ≈ vec3(0,0,0)
    @test JuliaGLM.faceforward(vec3(1,0,0), vec3(1,0,0), vec3(1,0,0)) == vec3(-1,0,0)
    @test JuliaGLM.faceforward(vec3(1,0,0), vec3(0,-1,0), vec3(0,1,0)) == vec3(1,0,0)
    @test JuliaGLM.reflect(vec3(1,-1,0), vec3(0,1,0)) ≈ vec3(1,1,0)
    @test JuliaGLM.reflect(vec3(1,1,0), vec3(1,0,1) ./ Float32(√2)) ≈ vec3(0,1,-1)
    @test JuliaGLM.refract(vec3(0,-1,0), vec3(0,1,0), 1.0f0) == vec3(0,-1,0)
    @test JuliaGLM.refract(vec3(1/√2,-1/√2,0), vec3(0,1,0), 1.0f0) == vec3(1/√2,-1/√2,0)
    @test JuliaGLM.refract(vec3(1/√2,-1/√2,0), vec3(0,1,0), 2.0f0) == vec3(0)
end

@testset "Relational Fns   " begin
    @test JuliaGLM.lessThan(vec3(0,1,2), vec3(2,1,1)) == bvec3(true,false,false)
    @test JuliaGLM.lessThanEqual(vec3(0,1,2), vec3(2,1,1)) == bvec3(true,true,false)
    @test JuliaGLM.greaterThan(vec3(0,1,2), vec3(2,1,1)) == bvec3(false,false,true)
    @test JuliaGLM.greaterThanEqual(vec3(0,1,2), vec3(2,1,1)) == bvec3(false,true,true)
    @test JuliaGLM.equal(vec3(1,1,2), vec3(1,3,1)) == bvec3(true,false,false)
    @test JuliaGLM.notEqual(vec3(1,1,2), vec3(1,3,1)) == bvec3(false,true,true)
    @test JuliaGLM.any(bvec2(true,false))
    @test !JuliaGLM.any(bvec2(false))
    @test JuliaGLM.all(bvec2(true))
    @test !JuliaGLM.all(bvec2(true,false))
    @test JuliaGLM.not(bvec2(true,false)) == bvec2(false,true)
end