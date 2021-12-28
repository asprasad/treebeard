; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@lookupTable = private global [1430 x [256 x i8]] undef
@model = private global [82000 x { <8 x float>, <8 x i16>, i16 }] undef
@offsets = private global [100 x i64] undef
@lengths = private global [100 x i64] undef

declare i8* @malloc(i64)

declare void @free(i8*)

define { float*, float*, i64, [1 x i64], [1 x i64] } @Prediction_Function(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, float* %7, float* %8, i64 %9, i64 %10, i64 %11) {
  %13 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %14 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %13, float* %1, 1
  %15 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %14, i64 %2, 2
  %16 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %15, i64 %3, 3, 0
  %17 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %16, i64 %5, 4, 0
  %18 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %17, i64 %4, 3, 1
  %19 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %18, i64 %6, 4, 1
  %20 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %7, 0
  %21 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %20, float* %8, 1
  %22 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %21, i64 %9, 2
  %23 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %22, i64 %10, 3, 0
  %24 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %23, i64 %11, 4, 0
  %25 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %26 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %25, float* %1, 1
  %27 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %26, i64 0, 2
  %28 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, i64 28, 3, 1
  %29 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, i64 1, 4, 1
  %30 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %29, i64 200, 3, 0
  %31 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %30, i64 28, 4, 0
  br label %32

32:                                               ; preds = %105, %12
  %33 = phi i64 [ 0, %12 ], [ %111, %105 ]
  %34 = icmp slt i64 %33, 200
  br i1 %34, label %35, label %112

35:                                               ; preds = %32
  %36 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %37 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %36, float* %1, 1
  %38 = mul i64 %33, 28
  %39 = add i64 0, %38
  %40 = add i64 %39, 0
  %41 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %37, i64 %40, 2
  %42 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %41, i64 28, 3, 1
  %43 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %42, i64 1, 4, 1
  %44 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %43, i64 1, 3, 0
  %45 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %44, i64 28, 4, 0
  br label %46

46:                                               ; preds = %98, %35
  %47 = phi i64 [ 0, %35 ], [ %104, %98 ]
  %48 = phi float [ -0.000000e+00, %35 ], [ %103, %98 ]
  %49 = icmp slt i64 %47, 100
  br i1 %49, label %50, label %105

50:                                               ; preds = %46
  %51 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @offsets, i64 0, i64 0), i64 %47
  %52 = load i64, i64* %51, align 4
  %53 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @lengths, i64 0, i64 0), i64 %47
  %54 = load i64, i64* %53, align 4
  %55 = mul i64 %52, 1
  %56 = add i64 0, %55
  %57 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16 }*), { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %56, 2
  %58 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } %57, i64 %54, 3, 0
  %59 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } %58, i64 1, 4, 0
  br label %60

60:                                               ; preds = %68, %50
  %61 = phi i64 [ %97, %68 ], [ 0, %50 ]
  %62 = add i64 %56, %61
  %63 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %62, i32 1
  %64 = load <8 x i16>, <8 x i16>* %63, align 16
  %65 = extractelement <8 x i16> %64, i32 0
  %66 = icmp eq i16 %65, -1
  %67 = icmp eq i1 %66, false
  br i1 %67, label %68, label %98

68:                                               ; preds = %60
  %69 = phi i64 [ %61, %60 ]
  %70 = add i64 %56, %69
  %71 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %70, i32 0
  %72 = load <8 x float>, <8 x float>* %71, align 32
  %73 = add i64 %56, %69
  %74 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %73, i32 1
  %75 = load <8 x i16>, <8 x i16>* %74, align 16
  %76 = add i64 %56, %69
  %77 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %76, i32 2
  %78 = load i16, i16* %77, align 2
  %79 = sext i16 %78 to i64
  %80 = mul i64 %69, 9
  %81 = add i64 %80, 1
  %82 = sext <8 x i16> %75 to <8 x i64>
  %83 = add i64 %40, 0
  %84 = add i64 %83, 0
  %85 = getelementptr float, float* %1, i64 %84
  %86 = getelementptr float, float* %85, <8 x i64> %82
  %87 = call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %86, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> zeroinitializer)
  %88 = fcmp ule <8 x float> %87, %72
  %89 = bitcast <8 x i1> %88 to <1 x i8>
  %90 = extractelement <1 x i8> %89, i32 0
  %91 = zext i8 %90 to i64
  %92 = mul i64 %79, 256
  %93 = add i64 %92, %91
  %94 = getelementptr i8, i8* getelementptr inbounds ([1430 x [256 x i8]], [1430 x [256 x i8]]* @lookupTable, i64 0, i64 0, i64 0), i64 %93
  %95 = load i8, i8* %94, align 1
  %96 = sext i8 %95 to i64
  %97 = add i64 %81, %96
  br label %60

98:                                               ; preds = %60
  %99 = add i64 %56, %61
  %100 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %99, i32 0
  %101 = load <8 x float>, <8 x float>* %100, align 32
  %102 = extractelement <8 x float> %101, i32 0
  %103 = fadd float %48, %102
  %104 = add i64 %47, 1
  br label %46

105:                                              ; preds = %46
  %106 = fneg float %48
  %107 = call float @llvm.exp.f32(float %106)
  %108 = fadd float 1.000000e+00, %107
  %109 = fdiv float 1.000000e+00, %108
  %110 = getelementptr float, float* %8, i64 %33
  store float %109, float* %110, align 4
  %111 = add i64 %33, 1
  br label %32

112:                                              ; preds = %32
  ret { float*, float*, i64, [1 x i64], [1 x i64] } %24
}

define { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } @Get_model() {
  ret { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16 }*), { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 0, [1 x i64] [i64 82000], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_offsets() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([100 x i64], [100 x i64]* @offsets, i64 0, i64 0), i64 0, [1 x i64] [i64 100], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_lengths() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([100 x i64], [100 x i64]* @lengths, i64 0, i64 0), i64 0, [1 x i64] [i64 100], [1 x i64] [i64 1] }
}

define i32 @Init_model(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i16* %5, i16* %6, i64 %7, i64 %8, i64 %9, i16* %10, i16* %11, i64 %12, i64 %13, i64 %14) {
  %16 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %0, 0
  %17 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %16, float* %1, 1
  %18 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } undef, i16* %5, 0
  %22 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %21, i16* %6, 1
  %23 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } undef, i16* %10, 0
  %27 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %26, i16* %11, 1
  %28 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %27, i64 %12, 2
  %29 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %28, i64 %13, 3, 0
  %30 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %29, i64 %14, 4, 0
  br label %31

31:                                               ; preds = %34, %15
  %32 = phi i64 [ 0, %15 ], [ %99, %34 ]
  %33 = icmp slt i64 %32, 82000
  br i1 %33, label %34, label %100

34:                                               ; preds = %31
  %35 = mul i64 %32, 8
  %36 = getelementptr float, float* %1, i64 %35
  %37 = load float, float* %36, align 4
  %38 = insertelement <8 x float> zeroinitializer, float %37, i32 0
  %39 = getelementptr i16, i16* %6, i64 %35
  %40 = load i16, i16* %39, align 2
  %41 = insertelement <8 x i16> zeroinitializer, i16 %40, i32 0
  %42 = add i64 %35, 1
  %43 = getelementptr float, float* %1, i64 %42
  %44 = load float, float* %43, align 4
  %45 = insertelement <8 x float> %38, float %44, i32 1
  %46 = getelementptr i16, i16* %6, i64 %42
  %47 = load i16, i16* %46, align 2
  %48 = insertelement <8 x i16> %41, i16 %47, i32 1
  %49 = add i64 %35, 2
  %50 = getelementptr float, float* %1, i64 %49
  %51 = load float, float* %50, align 4
  %52 = insertelement <8 x float> %45, float %51, i32 2
  %53 = getelementptr i16, i16* %6, i64 %49
  %54 = load i16, i16* %53, align 2
  %55 = insertelement <8 x i16> %48, i16 %54, i32 2
  %56 = add i64 %35, 3
  %57 = getelementptr float, float* %1, i64 %56
  %58 = load float, float* %57, align 4
  %59 = insertelement <8 x float> %52, float %58, i32 3
  %60 = getelementptr i16, i16* %6, i64 %56
  %61 = load i16, i16* %60, align 2
  %62 = insertelement <8 x i16> %55, i16 %61, i32 3
  %63 = add i64 %35, 4
  %64 = getelementptr float, float* %1, i64 %63
  %65 = load float, float* %64, align 4
  %66 = insertelement <8 x float> %59, float %65, i32 4
  %67 = getelementptr i16, i16* %6, i64 %63
  %68 = load i16, i16* %67, align 2
  %69 = insertelement <8 x i16> %62, i16 %68, i32 4
  %70 = add i64 %35, 5
  %71 = getelementptr float, float* %1, i64 %70
  %72 = load float, float* %71, align 4
  %73 = insertelement <8 x float> %66, float %72, i32 5
  %74 = getelementptr i16, i16* %6, i64 %70
  %75 = load i16, i16* %74, align 2
  %76 = insertelement <8 x i16> %69, i16 %75, i32 5
  %77 = add i64 %35, 6
  %78 = getelementptr float, float* %1, i64 %77
  %79 = load float, float* %78, align 4
  %80 = insertelement <8 x float> %73, float %79, i32 6
  %81 = getelementptr i16, i16* %6, i64 %77
  %82 = load i16, i16* %81, align 2
  %83 = insertelement <8 x i16> %76, i16 %82, i32 6
  %84 = add i64 %35, 7
  %85 = getelementptr float, float* %1, i64 %84
  %86 = load float, float* %85, align 4
  %87 = insertelement <8 x float> %80, float %86, i32 7
  %88 = getelementptr i16, i16* %6, i64 %84
  %89 = load i16, i16* %88, align 2
  %90 = insertelement <8 x i16> %83, i16 %89, i32 7
  %91 = getelementptr i16, i16* %11, i64 %32
  %92 = load i16, i16* %91, align 2
  %93 = add i64 0, %32
  %94 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %93, i32 0
  store <8 x float> %87, <8 x float>* %94, align 32
  %95 = add i64 0, %32
  %96 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %95, i32 1
  store <8 x i16> %90, <8 x i16>* %96, align 16
  %97 = add i64 0, %32
  %98 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %97, i32 2
  store i16 %92, i16* %98, align 2
  %99 = add i64 %32, 1
  br label %31

100:                                              ; preds = %31
  ret i32 trunc (i64 sub (i64 ptrtoint ({ <8 x float>, <8 x i16>, i16 }* getelementptr ({ <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 82000) to i64), i64 ptrtoint ([82000 x { <8 x float>, <8 x i16>, i16 }]* @model to i64)) to i32)
}

define { i8*, i8*, i64, [2 x i64], [2 x i64] } @Get_lookupTable() {
  ret { i8*, i8*, i64, [2 x i64], [2 x i64] } { i8* inttoptr (i64 3735928559 to i8*), i8* getelementptr inbounds ([1430 x [256 x i8]], [1430 x [256 x i8]]* @lookupTable, i64 0, i64 0, i64 0), i64 0, [2 x i64] [i64 1430, i64 256], [2 x i64] [i64 256, i64 1] }
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.exp.f32(float) #0

; Function Attrs: nofree nosync nounwind readonly willreturn
declare <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*>, i32 immarg, <8 x i1>, <8 x float>) #1

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #1 = { nofree nosync nounwind readonly willreturn }
