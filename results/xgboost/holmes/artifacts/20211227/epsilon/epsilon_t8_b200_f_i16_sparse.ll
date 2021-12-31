; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@lookupTable = private global [1430 x [256 x i8]] undef
@model = private global [9100 x { <8 x float>, <8 x i16>, i16, i16 }] undef
@offsets = private global [100 x i64] undef
@lengths = private global [100 x i64] undef
@leaves = private global [69138 x float] undef
@leavesOffsets = private global [100 x i64] undef
@leavesLengths = private global [100 x i64] undef

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
  %28 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, i64 2000, 3, 1
  %29 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, i64 1, 4, 1
  %30 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %29, i64 200, 3, 0
  %31 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %30, i64 2000, 4, 0
  br label %32

32:                                               ; preds = %132, %12
  %33 = phi i64 [ 0, %12 ], [ %138, %132 ]
  %34 = icmp slt i64 %33, 200
  br i1 %34, label %35, label %139

35:                                               ; preds = %32
  %36 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %37 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %36, float* %1, 1
  %38 = mul i64 %33, 2000
  %39 = add i64 0, %38
  %40 = add i64 %39, 0
  %41 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %37, i64 %40, 2
  %42 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %41, i64 2000, 3, 1
  %43 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %42, i64 1, 4, 1
  %44 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %43, i64 1, 3, 0
  %45 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %44, i64 2000, 4, 0
  br label %46

46:                                               ; preds = %129, %35
  %47 = phi i64 [ 0, %35 ], [ %131, %129 ]
  %48 = phi float [ -0.000000e+00, %35 ], [ %130, %129 ]
  %49 = icmp slt i64 %47, 100
  br i1 %49, label %50, label %132

50:                                               ; preds = %46
  %51 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @offsets, i64 0, i64 0), i64 %47
  %52 = load i64, i64* %51, align 4
  %53 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @lengths, i64 0, i64 0), i64 %47
  %54 = load i64, i64* %53, align 4
  %55 = mul i64 %52, 1
  %56 = add i64 0, %55
  %57 = insertvalue { { <8 x float>, <8 x i16>, i16, i16 }*, { <8 x float>, <8 x i16>, i16, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16, i16 }*), { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %56, 2
  %58 = insertvalue { { <8 x float>, <8 x i16>, i16, i16 }*, { <8 x float>, <8 x i16>, i16, i16 }*, i64, [1 x i64], [1 x i64] } %57, i64 %54, 3, 0
  %59 = insertvalue { { <8 x float>, <8 x i16>, i16, i16 }*, { <8 x float>, <8 x i16>, i16, i16 }*, i64, [1 x i64], [1 x i64] } %58, i64 1, 4, 0
  %60 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @leavesOffsets, i64 0, i64 0), i64 %47
  %61 = load i64, i64* %60, align 4
  %62 = getelementptr i64, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @leavesLengths, i64 0, i64 0), i64 %47
  %63 = load i64, i64* %62, align 4
  %64 = mul i64 %61, 1
  %65 = add i64 0, %64
  %66 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } { float* inttoptr (i64 3735928559 to float*), float* getelementptr inbounds ([69138 x float], [69138 x float]* @leaves, i64 0, i64 0), i64 undef, [1 x i64] undef, [1 x i64] undef }, i64 %65, 2
  %67 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %66, i64 %63, 3, 0
  %68 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %67, i64 1, 4, 0
  br label %69

69:                                               ; preds = %83, %50
  %70 = phi i64 [ %114, %83 ], [ 0, %50 ]
  %71 = icmp sge i64 %70, %54
  br i1 %71, label %72, label %73

72:                                               ; preds = %69
  br label %79

73:                                               ; preds = %69
  %74 = add i64 %56, %70
  %75 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %74, i32 1
  %76 = load <8 x i16>, <8 x i16>* %75, align 16
  %77 = extractelement <8 x i16> %76, i32 0
  %78 = icmp eq i16 %77, -1
  br label %79

79:                                               ; preds = %72, %73
  %80 = phi i1 [ %78, %73 ], [ true, %72 ]
  br label %81

81:                                               ; preds = %79
  %82 = icmp eq i1 %80, false
  br i1 %82, label %83, label %115

83:                                               ; preds = %81
  %84 = phi i64 [ %70, %81 ]
  %85 = add i64 %56, %84
  %86 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %85, i32 0
  %87 = load <8 x float>, <8 x float>* %86, align 32
  %88 = add i64 %56, %84
  %89 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %88, i32 1
  %90 = load <8 x i16>, <8 x i16>* %89, align 16
  %91 = add i64 %56, %84
  %92 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %91, i32 2
  %93 = load i16, i16* %92, align 2
  %94 = sext i16 %93 to i64
  %95 = add i64 %56, %84
  %96 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %95, i32 3
  %97 = load i16, i16* %96, align 2
  %98 = sext i16 %97 to i64
  %99 = sext <8 x i16> %90 to <8 x i64>
  %100 = add i64 %40, 0
  %101 = add i64 %100, 0
  %102 = getelementptr float, float* %1, i64 %101
  %103 = getelementptr float, float* %102, <8 x i64> %99
  %104 = call <8 x float> @llvm.masked.gather.v8f32.v8p0f32(<8 x float*> %103, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x float> zeroinitializer)
  %105 = fcmp ule <8 x float> %104, %87
  %106 = bitcast <8 x i1> %105 to <1 x i8>
  %107 = extractelement <1 x i8> %106, i32 0
  %108 = zext i8 %107 to i64
  %109 = mul i64 %94, 256
  %110 = add i64 %109, %108
  %111 = getelementptr i8, i8* getelementptr inbounds ([1430 x [256 x i8]], [1430 x [256 x i8]]* @lookupTable, i64 0, i64 0, i64 0), i64 %110
  %112 = load i8, i8* %111, align 1
  %113 = sext i8 %112 to i64
  %114 = add i64 %98, %113
  br label %69

115:                                              ; preds = %81
  %116 = icmp slt i64 %70, %54
  br i1 %116, label %117, label %122

117:                                              ; preds = %115
  %118 = add i64 %56, %70
  %119 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %118, i32 0
  %120 = load <8 x float>, <8 x float>* %119, align 32
  %121 = extractelement <8 x float> %120, i32 0
  br label %127

122:                                              ; preds = %115
  %123 = sub i64 %70, %54
  %124 = add i64 %65, %123
  %125 = getelementptr float, float* getelementptr inbounds ([69138 x float], [69138 x float]* @leaves, i64 0, i64 0), i64 %124
  %126 = load float, float* %125, align 4
  br label %127

127:                                              ; preds = %117, %122
  %128 = phi float [ %126, %122 ], [ %121, %117 ]
  br label %129

129:                                              ; preds = %127
  %130 = fadd float %48, %128
  %131 = add i64 %47, 1
  br label %46

132:                                              ; preds = %46
  %133 = fneg float %48
  %134 = call float @llvm.exp.f32(float %133)
  %135 = fadd float 1.000000e+00, %134
  %136 = fdiv float 1.000000e+00, %135
  %137 = getelementptr float, float* %8, i64 %33
  store float %136, float* %137, align 4
  %138 = add i64 %33, 1
  br label %32

139:                                              ; preds = %32
  ret { float*, float*, i64, [1 x i64], [1 x i64] } %24
}

define { { <8 x float>, <8 x i16>, i16, i16 }*, { <8 x float>, <8 x i16>, i16, i16 }*, i64, [1 x i64], [1 x i64] } @Get_model() {
  ret { { <8 x float>, <8 x i16>, i16, i16 }*, { <8 x float>, <8 x i16>, i16, i16 }*, i64, [1 x i64], [1 x i64] } { { <8 x float>, <8 x i16>, i16, i16 }* inttoptr (i64 3735928559 to { <8 x float>, <8 x i16>, i16, i16 }*), { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 0, [1 x i64] [i64 9100], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_offsets() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([100 x i64], [100 x i64]* @offsets, i64 0, i64 0), i64 0, [1 x i64] [i64 100], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_lengths() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([100 x i64], [100 x i64]* @lengths, i64 0, i64 0), i64 0, [1 x i64] [i64 100], [1 x i64] [i64 1] }
}

define { float*, float*, i64, [1 x i64], [1 x i64] } @Get_leaves() {
  ret { float*, float*, i64, [1 x i64], [1 x i64] } { float* inttoptr (i64 3735928559 to float*), float* getelementptr inbounds ([69138 x float], [69138 x float]* @leaves, i64 0, i64 0), i64 0, [1 x i64] [i64 69138], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_leavesOffsets() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([100 x i64], [100 x i64]* @leavesOffsets, i64 0, i64 0), i64 0, [1 x i64] [i64 100], [1 x i64] [i64 1] }
}

define { i64*, i64*, i64, [1 x i64], [1 x i64] } @Get_leavesLengths() {
  ret { i64*, i64*, i64, [1 x i64], [1 x i64] } { i64* inttoptr (i64 3735928559 to i64*), i64* getelementptr inbounds ([100 x i64], [100 x i64]* @leavesLengths, i64 0, i64 0), i64 0, [1 x i64] [i64 100], [1 x i64] [i64 1] }
}

define i32 @Init_model(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i16* %5, i16* %6, i64 %7, i64 %8, i64 %9, i16* %10, i16* %11, i64 %12, i64 %13, i64 %14, i16* %15, i16* %16, i64 %17, i64 %18, i64 %19) {
  %21 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %0, 0
  %22 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %21, float* %1, 1
  %23 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %22, i64 %2, 2
  %24 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %23, i64 %3, 3, 0
  %25 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %24, i64 %4, 4, 0
  %26 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } undef, i16* %5, 0
  %27 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %26, i16* %6, 1
  %28 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %27, i64 %7, 2
  %29 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %28, i64 %8, 3, 0
  %30 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %29, i64 %9, 4, 0
  %31 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } undef, i16* %10, 0
  %32 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %31, i16* %11, 1
  %33 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %32, i64 %12, 2
  %34 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %33, i64 %13, 3, 0
  %35 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %34, i64 %14, 4, 0
  %36 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } undef, i16* %15, 0
  %37 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %36, i16* %16, 1
  %38 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %37, i64 %17, 2
  %39 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %38, i64 %18, 3, 0
  %40 = insertvalue { i16*, i16*, i64, [1 x i64], [1 x i64] } %39, i64 %19, 4, 0
  br label %41

41:                                               ; preds = %44, %20
  %42 = phi i64 [ 0, %20 ], [ %113, %44 ]
  %43 = icmp slt i64 %42, 9100
  br i1 %43, label %44, label %114

44:                                               ; preds = %41
  %45 = mul i64 %42, 8
  %46 = getelementptr float, float* %1, i64 %45
  %47 = load float, float* %46, align 4
  %48 = insertelement <8 x float> zeroinitializer, float %47, i32 0
  %49 = getelementptr i16, i16* %6, i64 %45
  %50 = load i16, i16* %49, align 2
  %51 = insertelement <8 x i16> zeroinitializer, i16 %50, i32 0
  %52 = add i64 %45, 1
  %53 = getelementptr float, float* %1, i64 %52
  %54 = load float, float* %53, align 4
  %55 = insertelement <8 x float> %48, float %54, i32 1
  %56 = getelementptr i16, i16* %6, i64 %52
  %57 = load i16, i16* %56, align 2
  %58 = insertelement <8 x i16> %51, i16 %57, i32 1
  %59 = add i64 %45, 2
  %60 = getelementptr float, float* %1, i64 %59
  %61 = load float, float* %60, align 4
  %62 = insertelement <8 x float> %55, float %61, i32 2
  %63 = getelementptr i16, i16* %6, i64 %59
  %64 = load i16, i16* %63, align 2
  %65 = insertelement <8 x i16> %58, i16 %64, i32 2
  %66 = add i64 %45, 3
  %67 = getelementptr float, float* %1, i64 %66
  %68 = load float, float* %67, align 4
  %69 = insertelement <8 x float> %62, float %68, i32 3
  %70 = getelementptr i16, i16* %6, i64 %66
  %71 = load i16, i16* %70, align 2
  %72 = insertelement <8 x i16> %65, i16 %71, i32 3
  %73 = add i64 %45, 4
  %74 = getelementptr float, float* %1, i64 %73
  %75 = load float, float* %74, align 4
  %76 = insertelement <8 x float> %69, float %75, i32 4
  %77 = getelementptr i16, i16* %6, i64 %73
  %78 = load i16, i16* %77, align 2
  %79 = insertelement <8 x i16> %72, i16 %78, i32 4
  %80 = add i64 %45, 5
  %81 = getelementptr float, float* %1, i64 %80
  %82 = load float, float* %81, align 4
  %83 = insertelement <8 x float> %76, float %82, i32 5
  %84 = getelementptr i16, i16* %6, i64 %80
  %85 = load i16, i16* %84, align 2
  %86 = insertelement <8 x i16> %79, i16 %85, i32 5
  %87 = add i64 %45, 6
  %88 = getelementptr float, float* %1, i64 %87
  %89 = load float, float* %88, align 4
  %90 = insertelement <8 x float> %83, float %89, i32 6
  %91 = getelementptr i16, i16* %6, i64 %87
  %92 = load i16, i16* %91, align 2
  %93 = insertelement <8 x i16> %86, i16 %92, i32 6
  %94 = add i64 %45, 7
  %95 = getelementptr float, float* %1, i64 %94
  %96 = load float, float* %95, align 4
  %97 = insertelement <8 x float> %90, float %96, i32 7
  %98 = getelementptr i16, i16* %6, i64 %94
  %99 = load i16, i16* %98, align 2
  %100 = insertelement <8 x i16> %93, i16 %99, i32 7
  %101 = getelementptr i16, i16* %11, i64 %42
  %102 = load i16, i16* %101, align 2
  %103 = getelementptr i16, i16* %16, i64 %42
  %104 = load i16, i16* %103, align 2
  %105 = add i64 0, %42
  %106 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %105, i32 0
  store <8 x float> %97, <8 x float>* %106, align 32
  %107 = add i64 0, %42
  %108 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %107, i32 1
  store <8 x i16> %100, <8 x i16>* %108, align 16
  %109 = add i64 0, %42
  %110 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %109, i32 2
  store i16 %102, i16* %110, align 2
  %111 = add i64 0, %42
  %112 = getelementptr { <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 %111, i32 3
  store i16 %104, i16* %112, align 2
  %113 = add i64 %42, 1
  br label %41

114:                                              ; preds = %41
  ret i32 trunc (i64 sub (i64 ptrtoint ({ <8 x float>, <8 x i16>, i16, i16 }* getelementptr ({ <8 x float>, <8 x i16>, i16, i16 }, { <8 x float>, <8 x i16>, i16, i16 }* getelementptr inbounds ([9100 x { <8 x float>, <8 x i16>, i16, i16 }], [9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model, i64 0, i64 0), i64 9100) to i64), i64 ptrtoint ([9100 x { <8 x float>, <8 x i16>, i16, i16 }]* @model to i64)) to i32)
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
