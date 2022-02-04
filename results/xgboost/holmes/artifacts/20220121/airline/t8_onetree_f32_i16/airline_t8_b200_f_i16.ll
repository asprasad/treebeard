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
  %28 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, i64 13, 3, 1
  %29 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, i64 1, 4, 1
  %30 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %29, i64 200, 3, 0
  %31 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %30, i64 13, 4, 0
  br label %32

32:                                               ; preds = %35, %12
  %33 = phi i64 [ 0, %12 ], [ %37, %35 ]
  %34 = icmp slt i64 %33, 200
  br i1 %34, label %35, label %38

35:                                               ; preds = %32
  %36 = getelementptr float, float* %8, i64 %33
  store float -0.000000e+00, float* %36, align 4
  %37 = add i64 %33, 1
  br label %32

38:                                               ; preds = %32
  br label %39

39:                                               ; preds = %114, %38
  %40 = phi i64 [ 0, %38 ], [ %115, %114 ]
  %41 = icmp slt i64 %40, 100
  br i1 %41, label %42, label %116

42:                                               ; preds = %39
  %43 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @offsets, i64 0, i64 0), i64 %40
  %44 = load i64, i64* %43, align 4
  %45 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @lengths, i64 0, i64 0), i64 %40
  %46 = load i64, i64* %45, align 4
  %47 = mul i64 %44, 1
  %48 = add i64 0, %47
  %49 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16 }*), { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %48, 2
  %50 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } %49, i64 %46, 3, 0
  %51 = insertvalue { { <8 x float>, <8 x i16>, i16 }*, { <8 x float>, <8 x i16>, i16 }*, i64, [1 x i64], [1 x i64] } %50, i64 1, 4, 0
  br label %52

52:                                               ; preds = %104, %42
  %53 = phi i64 [ 0, %42 ], [ %113, %104 ]
  %54 = icmp slt i64 %53, 200
  br i1 %54, label %55, label %114

55:                                               ; preds = %52
  %56 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %57 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %56, float* %1, 1
  %58 = mul i64 %53, 13
  %59 = add i64 0, %58
  %60 = add i64 %59, 0
  %61 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %57, i64 %60, 2
  %62 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %61, i64 13, 3, 1
  %63 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %62, i64 1, 4, 1
  %64 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %63, i64 1, 3, 0
  %65 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %64, i64 13, 4, 0
  br label %66

66:                                               ; preds = %74, %55
  %67 = phi i64 [ %103, %74 ], [ 0, %55 ]
  %68 = add i64 %48, %67
  %69 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %68, i32 1
  %70 = load <8 x i16>, <8 x i16>* %69, align 16
  %71 = extractelement <8 x i16> %70, i32 0
  %72 = icmp eq i16 %71, -1
  %73 = icmp eq i1 %72, false
  br i1 %73, label %74, label %104

74:                                               ; preds = %66
  %75 = phi i64 [ %67, %66 ]
  %76 = add i64 %48, %75
  %77 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %76, i32 0
  %78 = load <8 x float>, <8 x float>* %77, align 32
  %79 = add i64 %48, %75
  %80 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %79, i32 1
  %81 = load <8 x i16>, <8 x i16>* %80, align 16
  %82 = add i64 %48, %75
  %83 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %82, i32 2
  %84 = load i16, i16* %83, align 2
  %85 = sext i16 %84 to i64
  %86 = mul i64 %75, 9
  %87 = add i64 %86, 1
  %88 = sext <8 x i16> %81 to <8 x i64>
  %89 = add i64 %60, 0
  %90 = add i64 %89, 0
  %91 = getelementptr float, float* %1, i64 %90
  %92 = getelementptr float, float* %91, <8 x i64> %88
  %93 = call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %92, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> zeroinitializer)
  %94 = fcmp ult <8 x float> %93, %78
  %95 = bitcast <8 x i1> %94 to <1 x i8>
  %96 = extractelement <1 x i8> %95, i32 0
  %97 = zext i8 %96 to i64
  %98 = mul i64 %85, 256
  %99 = add i64 %98, %97
  %100 = getelementptr i8, i8* getelementptr inbounds ([1430 x [256 x i8]], [1430 x [256 x i8]]* @lookupTable, i64 0, i64 0, i64 0), i64 %99
  %101 = load i8, i8* %100, align 1
  %102 = sext i8 %101 to i64
  %103 = add i64 %87, %102
  br label %66

104:                                              ; preds = %66
  %105 = add i64 %48, %67
  %106 = getelementptr { <8 x float>, <8 x i16>, i16 }, { <8 x float>, <8 x i16>, i16 }* getelementptr inbounds ([82000 x { <8 x float>, <8 x i16>, i16 }], [82000 x { <8 x float>, <8 x i16>, i16 }]* @model, i64 0, i64 0), i64 %105, i32 0
  %107 = load <8 x float>, <8 x float>* %106, align 32
  %108 = extractelement <8 x float> %107, i32 0
  %109 = getelementptr float, float* %8, i64 %53
  %110 = load float, float* %109, align 4
  %111 = fadd float %108, %110
  %112 = getelementptr float, float* %8, i64 %53
  store float %111, float* %112, align 4
  %113 = add i64 %53, 1
  br label %52

114:                                              ; preds = %52
  %115 = add i64 %40, 1
  br label %39

116:                                              ; preds = %39
  br label %117

117:                                              ; preds = %120, %116
  %118 = phi i64 [ 0, %116 ], [ %128, %120 ]
  %119 = icmp slt i64 %118, 200
  br i1 %119, label %120, label %129

120:                                              ; preds = %117
  %121 = getelementptr float, float* %8, i64 %118
  %122 = load float, float* %121, align 4
  %123 = fneg float %122
  %124 = call float @llvm.exp.f32(float %123)
  %125 = fadd float 1.000000e+00, %124
  %126 = fdiv float 1.000000e+00, %125
  %127 = getelementptr float, float* %8, i64 %118
  store float %126, float* %127, align 4
  %128 = add i64 %118, 1
  br label %117

129:                                              ; preds = %117
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
