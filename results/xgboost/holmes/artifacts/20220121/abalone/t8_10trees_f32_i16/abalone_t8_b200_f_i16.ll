; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@lookupTable = private global [1430 x [256 x i8]] undef
@model = private global [91000 x { <8 x float>, <8 x i16>, i16 }] undef
@offsets = private global [1000 x i64] undef
@lengths = private global [1000 x i64] undef

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
  %28 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, i64 8, 3, 1
  %29 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, i64 1, 4, 1
  %30 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %29, i64 200, 3, 0
  %31 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %30, i64 8, 4, 0
  br label %32

32:                                               ; preds = %35, %12
  %33 = phi i64 [ 0, %12 ], [ %37, %35 ]
  %34 = icmp slt i64 %33, 200
  br i1 %34, label %35, label %38

35:                                               ; preds = %32
  %36 = getelementptr float, float* %8, i64 %33
  store float 5.000000e-01, float* %36, align 4
  %37 = add i64 %33, 1
  br label %32

38:                                               ; preds = %32
  br label %39

39:                                               ; preds = %123, %38
  %40 = phi i64 [ 0, %38 ], [ %124, %123 ]
  %41 = icmp slt i64 %40, 1000
  br i1 %41, label %42, label %125

42:                                               ; preds = %39
  br label %43

43:                                               ; preds = %117, %42
  %44 = phi i64 [ 0, %42 ], [ %122, %117 ]
  %45 = icmp slt i64 %44, 200
  br i1 %45, label %46, label %123

46:                                               ; preds = %43
  %47 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %48 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %47, float* %1, 1
  %49 = mul i64 %44, 8
  %50 = add i64 0, %49
  %51 = add i64 %50, 0
  %52 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %48, i64 %51, 2
  %53 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %52, i64 8, 3, 1
  %54 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %53, i64 1, 4, 1
  %55 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %54, i64 1, 3, 0
  %56 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %55, i64 8, 4, 0
  br label %57

57:                                               ; preds = %110, %46
  %58 = phi i64 [ 0, %46 ], [ %116, %110 ]
  %59 = phi float [ 0.000000e+00, %46 ], [ %115, %110 ]
  %60 = icmp slt i64 %58, 10
  br i1 %60, label %61, label %117

61:                                               ; preds = %57
  %62 = add i64 %40, %58
  %63 = getelementptr i64, i64* getelementptr inbounds ([1000 x i64], [1000 x i64]* @offsets, i64 0, i64 0), i64 %62
  %64 = load i64, i64* %63, align 4
  %65 = getelementptr i64, i64* getelementptr inbounds ([1000 x i64], [1000 x i64]* @lengths, i64 0, i64 0), i64 %62
  %66 = load i64, i64* %65, align 4
  %67 = mul i64 %64, 1
  %68 = add i64 0, %67
  %69 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16 }*), { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %68, 2
  %70 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } %69, i64 %66, 3, 0
  %71 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } %70, i64 1, 4, 0
  br label %72

72:                                               ; preds = %80, %61
  %73 = phi i64 [ %109, %80 ], [ 0, %61 ]
  %74 = add i64 %68, %73
  %75 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %74, i32 1
  %76 = load <8 x i16>, <8 x i16>* %75, align 16
  %77 = extractelement <8 x i16> %76, i32 0
  %78 = icmp eq i16 %77, -1
  %79 = icmp eq i1 %78, false
  br i1 %79, label %80, label %110

80:                                               ; preds = %72
  %81 = phi i64 [ %73, %72 ]
  %82 = add i64 %68, %81
  %83 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %82, i32 0
  %84 = load <8 x float>, <8 x float>* %83, align 32
  %85 = add i64 %68, %81
  %86 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %85, i32 1
  %87 = load <8 x i16>, <8 x i16>* %86, align 16
  %88 = add i64 %68, %81
  %89 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %88, i32 2
  %90 = load i16, i16* %89, align 2
  %91 = sext i16 %90 to i64
  %92 = mul i64 %81, 9
  %93 = add i64 %92, 1
  %94 = sext <8 x i16> %87 to <8 x i64>
  %95 = add i64 %51, 0
  %96 = add i64 %95, 0
  %97 = getelementptr float, float* %1, i64 %96
  %98 = getelementptr float, float* %97, <8 x i64> %94
  %99 = call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %98, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> zeroinitializer)
  %100 = fcmp ult <8 x float> %99, %84
  %101 = bitcast <8 x i1> %100 to <1 x i8>
  %102 = extractelement <1 x i8> %101, i32 0
  %103 = zext i8 %102 to i64
  %104 = mul i64 %91, 256
  %105 = add i64 %104, %103
  %106 = getelementptr i8, i8* getelementptr inbounds ([1430 x [256 x i8]], [1430 x [256 x i8]]* @lookupTable, i64 0, i64 0, i64 0), i64 %105
  %107 = load i8, i8* %106, align 1
  %108 = sext i8 %107 to i64
  %109 = add i64 %93, %108
  br label %72

110:                                              ; preds = %72
  %111 = add i64 %68, %73
  %112 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %111, i32 0
  %113 = load <8 x float>, <8 x float>* %112, align 32
  %114 = extractelement <8 x float> %113, i32 0
  %115 = fadd float %59, %114
  %116 = add i64 %58, 1
  br label %57

117:                                              ; preds = %57
  %118 = getelementptr float, float* %8, i64 %44
  %119 = load float, float* %118, align 4
  %120 = fadd float %59, %119
  %121 = getelementptr float, float* %8, i64 %44
  store float %120, float* %121, align 4
  %122 = add i64 %44, 1
  br label %43

123:                                              ; preds = %43
  %124 = add i64 %40, 10
  br label %39

125:                                              ; preds = %39
  ret { float*, float*, i64, [1 x i64], [1 x i64] } %24
}

define { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } @Get_model() {
  ret { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16 }*), { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 0, [1 x i64] [i64 91000], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_offsets() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([1000 x i64], [1000 x i64]* @offsets, i64 0, i64 0), i64 0, [1 x i64] [i64 1000], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_lengths() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([1000 x i64], [1000 x i64]* @lengths, i64 0, i64 0), i64 0, [1 x i64] [i64 1000], [1 x i64] [i64 1] }
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
  %33 = icmp slt i64 %32, 91000
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
  %94 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %93, i32 0
  store <8 x float> %87, <8 x float>* %94, align 32
  %95 = add i64 0, %32
  %96 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %95, i32 1
  store <8 x i16> %90, <8 x i16>* %96, align 16
  %97 = add i64 0, %32
  %98 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %97, i32 2
  store i16 %92, i16* %98, align 2
  %99 = add i64 %32, 1
  br label %31

100:                                              ; preds = %31
  ret i32 trunc (i64 sub (i64 ptrtoint ({ <8 x float>, <8 x i16>, i16 }* getelementptr ({ <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([91000 x { <8 x float>, <8 x i16>, i16 }], [91000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 91000) to i64), i64 ptrtoint ([91000 x { <8 x float>, <8 x i16>, i16 }]* @model to i64)) to i32)
}

define { i8*, i8*, i64, [2 x i64], [2 x i64] } @Get_lookupTable() {
  ret { i8*, i8*, i64, [2 x i64], [2 x i64] } { i8* inttoptr (i64 3735928559 to i8*), i8* getelementptr inbounds ([1430 x [256 x i8]], [1430 x [256 x i8]]* @lookupTable, i64 0, i64 0, i64 0), i64 0, [2 x i64] [i64 1430, i64 256], [2 x i64] [i64 256, i64 1] }
}

; Function Attrs: nofree nosync nounwind readonly willreturn
declare <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*>, i32 immarg, <8 x i1>, <8 x float>) #0

attributes #0 = { nofree nosync nounwind readonly willreturn }
