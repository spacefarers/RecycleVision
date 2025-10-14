%main = fn(%input: f32[?,3,224,224]): // (f32[?,3,224,224]) -> f32[?,?]
{
  %0 = Reshape(const(i64[4] : {1L,1L,1L,1L}), const(i32[2] : {2,-1})): // i64[2,2]
  %1 = Transpose(%0, const(i32[2] : {1,0})): // i64[2,2]
  %2 = Conv2D(PadMode.Constant, %input, const(f32[16,3,3,3]), const(f32[16]), const(i64[2] : {2L,2L}), %1, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,16,?,?]
  %3 = HardSigmoid(%2, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,16,?,?]
  %4 = Binary(BinaryOp.Mul, %2, %3): // f32[?,?,?,?]
  %5 = Reshape(const(i64[4] : {1L,1L,1L,1L}), const(i32[2] : {2,-1})): // i64[2,2]
  %6 = Transpose(%5, const(i32[2] : {1,0})): // i64[2,2]
  %7 = Conv2D(PadMode.Constant, %4, const(f32[16,1,3,3]), const(f32[16]), const(i64[2] : {2L,2L}), %6, const(i64[2] : {1L,1L}), const(i64 : 16), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,16,?,?]
  %8 = Relu(%7): // f32[?,16,?,?]
  %9 = ShapeOf(%8): // i64[4]
  %10 = (const(i32 : 2)): // (i32)
  
  %11 = Stack(%10, const(i32 : 0)): // i32[1]
  %12 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %13 = (%12): // (i32)
  
  %14 = Stack(%13, const(i32 : 0)): // i32[1]
  %15 = Slice(%9, %11, %14, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %16 = Squeeze(%15, const(i64[1] : {0L})): // i64
  %17 = ShapeOf(%8): // i64[4]
  %18 = (const(i32 : 3)): // (i32)
  
  %19 = Stack(%18, const(i32 : 0)): // i32[1]
  %20 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %21 = (%20): // (i32)
  
  %22 = Stack(%21, const(i32 : 0)): // i32[1]
  %23 = Slice(%17, %19, %22, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %24 = Squeeze(%23, const(i64[1] : {0L})): // i64
  %25 = (%16, %24): // (i64, i64)
  
  %26 = Stack(%25, const(i32 : 0)): // i64[2]
  %27 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %28 = Transpose(%27, const(i32[2] : {1,0})): // i64[2,2]
  %29 = ReduceWindow2D(ReduceOp.Mean, %8, const(f32 : 0), %26, const(i64[2] : {1L,1L}), %28, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %30 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %31 = Transpose(%30, const(i32[2] : {1,0})): // i64[2,2]
  %32 = Conv2D(PadMode.Constant, %29, const(f32[8,16,1,1]), const(f32[8] : {-1.7193538E-09f,-2.3164939E-09f,-0.00016354494f,-1.8773305E-09f,-1.2890734E-09f,1.7531045E-08f,-5.6567337E-09f,9.671734E-11f}), const(i64[2] : {1L,1L}), %31, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,8,?,?]
  %33 = Relu(%32): // f32[?,8,?,?]
  %34 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %35 = Transpose(%34, const(i32[2] : {1,0})): // i64[2,2]
  %36 = Conv2D(PadMode.Constant, %33, const(f32[16,8,1,1]), const(f32[16]), const(i64[2] : {1L,1L}), %35, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,16,?,?]
  %37 = HardSigmoid(%36, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,16,?,?]
  %38 = Binary(BinaryOp.Mul, %37, %8): // f32[?,?,?,?]
  %39 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %40 = Transpose(%39, const(i32[2] : {1,0})): // i64[2,2]
  %41 = Conv2D(PadMode.Constant, %38, const(f32[16,16,1,1]), const(f32[16]), const(i64[2] : {1L,1L}), %40, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,16,?,?]
  %42 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %43 = Transpose(%42, const(i32[2] : {1,0})): // i64[2,2]
  %44 = Conv2D(PadMode.Constant, %41, const(f32[72,16,1,1]), const(f32[72]), const(i64[2] : {1L,1L}), %43, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,72,?,?]
  %45 = Relu(%44): // f32[?,72,?,?]
  %46 = Reshape(const(i64[4] : {1L,1L,1L,1L}), const(i32[2] : {2,-1})): // i64[2,2]
  %47 = Transpose(%46, const(i32[2] : {1,0})): // i64[2,2]
  %48 = Conv2D(PadMode.Constant, %45, const(f32[72,1,3,3]), const(f32[72]), const(i64[2] : {2L,2L}), %47, const(i64[2] : {1L,1L}), const(i64 : 72), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,72,?,?]
  %49 = Relu(%48): // f32[?,72,?,?]
  %50 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %51 = Transpose(%50, const(i32[2] : {1,0})): // i64[2,2]
  %52 = Conv2D(PadMode.Constant, %49, const(f32[24,72,1,1]), const(f32[24]), const(i64[2] : {1L,1L}), %51, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,24,?,?]
  %53 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %54 = Transpose(%53, const(i32[2] : {1,0})): // i64[2,2]
  %55 = Conv2D(PadMode.Constant, %52, const(f32[88,24,1,1]), const(f32[88]), const(i64[2] : {1L,1L}), %54, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,88,?,?]
  %56 = Relu(%55): // f32[?,88,?,?]
  %57 = Reshape(const(i64[4] : {1L,1L,1L,1L}), const(i32[2] : {2,-1})): // i64[2,2]
  %58 = Transpose(%57, const(i32[2] : {1,0})): // i64[2,2]
  %59 = Conv2D(PadMode.Constant, %56, const(f32[88,1,3,3]), const(f32[88]), const(i64[2] : {1L,1L}), %58, const(i64[2] : {1L,1L}), const(i64 : 88), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,88,?,?]
  %60 = Relu(%59): // f32[?,88,?,?]
  %61 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %62 = Transpose(%61, const(i32[2] : {1,0})): // i64[2,2]
  %63 = Conv2D(PadMode.Constant, %60, const(f32[24,88,1,1]), const(f32[24]), const(i64[2] : {1L,1L}), %62, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,24,?,?]
  %64 = Binary(BinaryOp.Add, %63, %52): // f32[?,?,?,?]
  %65 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %66 = Transpose(%65, const(i32[2] : {1,0})): // i64[2,2]
  %67 = Conv2D(PadMode.Constant, %64, const(f32[96,24,1,1]), const(f32[96]), const(i64[2] : {1L,1L}), %66, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,96,?,?]
  %68 = HardSigmoid(%67, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,96,?,?]
  %69 = Binary(BinaryOp.Mul, %67, %68): // f32[?,?,?,?]
  %70 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %71 = Transpose(%70, const(i32[2] : {1,0})): // i64[2,2]
  %72 = Conv2D(PadMode.Constant, %69, const(f32[96,1,5,5]), const(f32[96]), const(i64[2] : {2L,2L}), %71, const(i64[2] : {1L,1L}), const(i64 : 96), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,96,?,?]
  %73 = HardSigmoid(%72, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,96,?,?]
  %74 = Binary(BinaryOp.Mul, %72, %73): // f32[?,?,?,?]
  %75 = ShapeOf(%74): // i64[4]
  %76 = (const(i32 : 2)): // (i32)
  
  %77 = Stack(%76, const(i32 : 0)): // i32[1]
  %78 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %79 = (%78): // (i32)
  
  %80 = Stack(%79, const(i32 : 0)): // i32[1]
  %81 = Slice(%75, %77, %80, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %82 = Squeeze(%81, const(i64[1] : {0L})): // i64
  %83 = ShapeOf(%74): // i64[4]
  %84 = (const(i32 : 3)): // (i32)
  
  %85 = Stack(%84, const(i32 : 0)): // i32[1]
  %86 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %87 = (%86): // (i32)
  
  %88 = Stack(%87, const(i32 : 0)): // i32[1]
  %89 = Slice(%83, %85, %88, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %90 = Squeeze(%89, const(i64[1] : {0L})): // i64
  %91 = (%82, %90): // (i64, i64)
  
  %92 = Stack(%91, const(i32 : 0)): // i64[2]
  %93 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %94 = Transpose(%93, const(i32[2] : {1,0})): // i64[2,2]
  %95 = ReduceWindow2D(ReduceOp.Mean, %74, const(f32 : 0), %92, const(i64[2] : {1L,1L}), %94, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %96 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %97 = Transpose(%96, const(i32[2] : {1,0})): // i64[2,2]
  %98 = Conv2D(PadMode.Constant, %95, const(f32[24,96,1,1]), const(f32[24]), const(i64[2] : {1L,1L}), %97, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,24,?,?]
  %99 = Relu(%98): // f32[?,24,?,?]
  %100 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %101 = Transpose(%100, const(i32[2] : {1,0})): // i64[2,2]
  %102 = Conv2D(PadMode.Constant, %99, const(f32[96,24,1,1]), const(f32[96]), const(i64[2] : {1L,1L}), %101, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,96,?,?]
  %103 = HardSigmoid(%102, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,96,?,?]
  %104 = Binary(BinaryOp.Mul, %103, %74): // f32[?,?,?,?]
  %105 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %106 = Transpose(%105, const(i32[2] : {1,0})): // i64[2,2]
  %107 = Conv2D(PadMode.Constant, %104, const(f32[40,96,1,1]), const(f32[40]), const(i64[2] : {1L,1L}), %106, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,40,?,?]
  %108 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %109 = Transpose(%108, const(i32[2] : {1,0})): // i64[2,2]
  %110 = Conv2D(PadMode.Constant, %107, const(f32[240,40,1,1]), const(f32[240]), const(i64[2] : {1L,1L}), %109, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,240,?,?]
  %111 = HardSigmoid(%110, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,240,?,?]
  %112 = Binary(BinaryOp.Mul, %110, %111): // f32[?,?,?,?]
  %113 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %114 = Transpose(%113, const(i32[2] : {1,0})): // i64[2,2]
  %115 = Conv2D(PadMode.Constant, %112, const(f32[240,1,5,5]), const(f32[240]), const(i64[2] : {1L,1L}), %114, const(i64[2] : {1L,1L}), const(i64 : 240), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,240,?,?]
  %116 = HardSigmoid(%115, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,240,?,?]
  %117 = Binary(BinaryOp.Mul, %115, %116): // f32[?,?,?,?]
  %118 = ShapeOf(%117): // i64[4]
  %119 = (const(i32 : 2)): // (i32)
  
  %120 = Stack(%119, const(i32 : 0)): // i32[1]
  %121 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %122 = (%121): // (i32)
  
  %123 = Stack(%122, const(i32 : 0)): // i32[1]
  %124 = Slice(%118, %120, %123, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %125 = Squeeze(%124, const(i64[1] : {0L})): // i64
  %126 = ShapeOf(%117): // i64[4]
  %127 = (const(i32 : 3)): // (i32)
  
  %128 = Stack(%127, const(i32 : 0)): // i32[1]
  %129 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %130 = (%129): // (i32)
  
  %131 = Stack(%130, const(i32 : 0)): // i32[1]
  %132 = Slice(%126, %128, %131, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %133 = Squeeze(%132, const(i64[1] : {0L})): // i64
  %134 = (%125, %133): // (i64, i64)
  
  %135 = Stack(%134, const(i32 : 0)): // i64[2]
  %136 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %137 = Transpose(%136, const(i32[2] : {1,0})): // i64[2,2]
  %138 = ReduceWindow2D(ReduceOp.Mean, %117, const(f32 : 0), %135, const(i64[2] : {1L,1L}), %137, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %139 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %140 = Transpose(%139, const(i32[2] : {1,0})): // i64[2,2]
  %141 = Conv2D(PadMode.Constant, %138, const(f32[64,240,1,1]), const(f32[64]), const(i64[2] : {1L,1L}), %140, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,64,?,?]
  %142 = Relu(%141): // f32[?,64,?,?]
  %143 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %144 = Transpose(%143, const(i32[2] : {1,0})): // i64[2,2]
  %145 = Conv2D(PadMode.Constant, %142, const(f32[240,64,1,1]), const(f32[240]), const(i64[2] : {1L,1L}), %144, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,240,?,?]
  %146 = HardSigmoid(%145, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,240,?,?]
  %147 = Binary(BinaryOp.Mul, %146, %117): // f32[?,?,?,?]
  %148 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %149 = Transpose(%148, const(i32[2] : {1,0})): // i64[2,2]
  %150 = Conv2D(PadMode.Constant, %147, const(f32[40,240,1,1]), const(f32[40]), const(i64[2] : {1L,1L}), %149, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,40,?,?]
  %151 = Binary(BinaryOp.Add, %150, %107): // f32[?,?,?,?]
  %152 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %153 = Transpose(%152, const(i32[2] : {1,0})): // i64[2,2]
  %154 = Conv2D(PadMode.Constant, %151, const(f32[240,40,1,1]), const(f32[240]), const(i64[2] : {1L,1L}), %153, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,240,?,?]
  %155 = HardSigmoid(%154, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,240,?,?]
  %156 = Binary(BinaryOp.Mul, %154, %155): // f32[?,?,?,?]
  %157 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %158 = Transpose(%157, const(i32[2] : {1,0})): // i64[2,2]
  %159 = Conv2D(PadMode.Constant, %156, const(f32[240,1,5,5]), const(f32[240]), const(i64[2] : {1L,1L}), %158, const(i64[2] : {1L,1L}), const(i64 : 240), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,240,?,?]
  %160 = HardSigmoid(%159, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,240,?,?]
  %161 = Binary(BinaryOp.Mul, %159, %160): // f32[?,?,?,?]
  %162 = ShapeOf(%161): // i64[4]
  %163 = (const(i32 : 2)): // (i32)
  
  %164 = Stack(%163, const(i32 : 0)): // i32[1]
  %165 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %166 = (%165): // (i32)
  
  %167 = Stack(%166, const(i32 : 0)): // i32[1]
  %168 = Slice(%162, %164, %167, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %169 = Squeeze(%168, const(i64[1] : {0L})): // i64
  %170 = ShapeOf(%161): // i64[4]
  %171 = (const(i32 : 3)): // (i32)
  
  %172 = Stack(%171, const(i32 : 0)): // i32[1]
  %173 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %174 = (%173): // (i32)
  
  %175 = Stack(%174, const(i32 : 0)): // i32[1]
  %176 = Slice(%170, %172, %175, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %177 = Squeeze(%176, const(i64[1] : {0L})): // i64
  %178 = (%169, %177): // (i64, i64)
  
  %179 = Stack(%178, const(i32 : 0)): // i64[2]
  %180 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %181 = Transpose(%180, const(i32[2] : {1,0})): // i64[2,2]
  %182 = ReduceWindow2D(ReduceOp.Mean, %161, const(f32 : 0), %179, const(i64[2] : {1L,1L}), %181, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %183 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %184 = Transpose(%183, const(i32[2] : {1,0})): // i64[2,2]
  %185 = Conv2D(PadMode.Constant, %182, const(f32[64,240,1,1]), const(f32[64]), const(i64[2] : {1L,1L}), %184, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,64,?,?]
  %186 = Relu(%185): // f32[?,64,?,?]
  %187 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %188 = Transpose(%187, const(i32[2] : {1,0})): // i64[2,2]
  %189 = Conv2D(PadMode.Constant, %186, const(f32[240,64,1,1]), const(f32[240]), const(i64[2] : {1L,1L}), %188, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,240,?,?]
  %190 = HardSigmoid(%189, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,240,?,?]
  %191 = Binary(BinaryOp.Mul, %190, %161): // f32[?,?,?,?]
  %192 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %193 = Transpose(%192, const(i32[2] : {1,0})): // i64[2,2]
  %194 = Conv2D(PadMode.Constant, %191, const(f32[40,240,1,1]), const(f32[40]), const(i64[2] : {1L,1L}), %193, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,40,?,?]
  %195 = Binary(BinaryOp.Add, %194, %151): // f32[?,?,?,?]
  %196 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %197 = Transpose(%196, const(i32[2] : {1,0})): // i64[2,2]
  %198 = Conv2D(PadMode.Constant, %195, const(f32[120,40,1,1]), const(f32[120]), const(i64[2] : {1L,1L}), %197, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,120,?,?]
  %199 = HardSigmoid(%198, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,120,?,?]
  %200 = Binary(BinaryOp.Mul, %198, %199): // f32[?,?,?,?]
  %201 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %202 = Transpose(%201, const(i32[2] : {1,0})): // i64[2,2]
  %203 = Conv2D(PadMode.Constant, %200, const(f32[120,1,5,5]), const(f32[120]), const(i64[2] : {1L,1L}), %202, const(i64[2] : {1L,1L}), const(i64 : 120), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,120,?,?]
  %204 = HardSigmoid(%203, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,120,?,?]
  %205 = Binary(BinaryOp.Mul, %203, %204): // f32[?,?,?,?]
  %206 = ShapeOf(%205): // i64[4]
  %207 = (const(i32 : 2)): // (i32)
  
  %208 = Stack(%207, const(i32 : 0)): // i32[1]
  %209 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %210 = (%209): // (i32)
  
  %211 = Stack(%210, const(i32 : 0)): // i32[1]
  %212 = Slice(%206, %208, %211, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %213 = Squeeze(%212, const(i64[1] : {0L})): // i64
  %214 = ShapeOf(%205): // i64[4]
  %215 = (const(i32 : 3)): // (i32)
  
  %216 = Stack(%215, const(i32 : 0)): // i32[1]
  %217 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %218 = (%217): // (i32)
  
  %219 = Stack(%218, const(i32 : 0)): // i32[1]
  %220 = Slice(%214, %216, %219, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %221 = Squeeze(%220, const(i64[1] : {0L})): // i64
  %222 = (%213, %221): // (i64, i64)
  
  %223 = Stack(%222, const(i32 : 0)): // i64[2]
  %224 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %225 = Transpose(%224, const(i32[2] : {1,0})): // i64[2,2]
  %226 = ReduceWindow2D(ReduceOp.Mean, %205, const(f32 : 0), %223, const(i64[2] : {1L,1L}), %225, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %227 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %228 = Transpose(%227, const(i32[2] : {1,0})): // i64[2,2]
  %229 = Conv2D(PadMode.Constant, %226, const(f32[32,120,1,1]), const(f32[32]), const(i64[2] : {1L,1L}), %228, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,32,?,?]
  %230 = Relu(%229): // f32[?,32,?,?]
  %231 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %232 = Transpose(%231, const(i32[2] : {1,0})): // i64[2,2]
  %233 = Conv2D(PadMode.Constant, %230, const(f32[120,32,1,1]), const(f32[120]), const(i64[2] : {1L,1L}), %232, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,120,?,?]
  %234 = HardSigmoid(%233, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,120,?,?]
  %235 = Binary(BinaryOp.Mul, %234, %205): // f32[?,?,?,?]
  %236 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %237 = Transpose(%236, const(i32[2] : {1,0})): // i64[2,2]
  %238 = Conv2D(PadMode.Constant, %235, const(f32[48,120,1,1]), const(f32[48]), const(i64[2] : {1L,1L}), %237, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,48,?,?]
  %239 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %240 = Transpose(%239, const(i32[2] : {1,0})): // i64[2,2]
  %241 = Conv2D(PadMode.Constant, %238, const(f32[144,48,1,1]), const(f32[144]), const(i64[2] : {1L,1L}), %240, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,144,?,?]
  %242 = HardSigmoid(%241, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,144,?,?]
  %243 = Binary(BinaryOp.Mul, %241, %242): // f32[?,?,?,?]
  %244 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %245 = Transpose(%244, const(i32[2] : {1,0})): // i64[2,2]
  %246 = Conv2D(PadMode.Constant, %243, const(f32[144,1,5,5]), const(f32[144]), const(i64[2] : {1L,1L}), %245, const(i64[2] : {1L,1L}), const(i64 : 144), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,144,?,?]
  %247 = HardSigmoid(%246, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,144,?,?]
  %248 = Binary(BinaryOp.Mul, %246, %247): // f32[?,?,?,?]
  %249 = ShapeOf(%248): // i64[4]
  %250 = (const(i32 : 2)): // (i32)
  
  %251 = Stack(%250, const(i32 : 0)): // i32[1]
  %252 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %253 = (%252): // (i32)
  
  %254 = Stack(%253, const(i32 : 0)): // i32[1]
  %255 = Slice(%249, %251, %254, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %256 = Squeeze(%255, const(i64[1] : {0L})): // i64
  %257 = ShapeOf(%248): // i64[4]
  %258 = (const(i32 : 3)): // (i32)
  
  %259 = Stack(%258, const(i32 : 0)): // i32[1]
  %260 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %261 = (%260): // (i32)
  
  %262 = Stack(%261, const(i32 : 0)): // i32[1]
  %263 = Slice(%257, %259, %262, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %264 = Squeeze(%263, const(i64[1] : {0L})): // i64
  %265 = (%256, %264): // (i64, i64)
  
  %266 = Stack(%265, const(i32 : 0)): // i64[2]
  %267 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %268 = Transpose(%267, const(i32[2] : {1,0})): // i64[2,2]
  %269 = ReduceWindow2D(ReduceOp.Mean, %248, const(f32 : 0), %266, const(i64[2] : {1L,1L}), %268, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %270 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %271 = Transpose(%270, const(i32[2] : {1,0})): // i64[2,2]
  %272 = Conv2D(PadMode.Constant, %269, const(f32[40,144,1,1]), const(f32[40]), const(i64[2] : {1L,1L}), %271, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,40,?,?]
  %273 = Relu(%272): // f32[?,40,?,?]
  %274 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %275 = Transpose(%274, const(i32[2] : {1,0})): // i64[2,2]
  %276 = Conv2D(PadMode.Constant, %273, const(f32[144,40,1,1]), const(f32[144]), const(i64[2] : {1L,1L}), %275, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,144,?,?]
  %277 = HardSigmoid(%276, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,144,?,?]
  %278 = Binary(BinaryOp.Mul, %277, %248): // f32[?,?,?,?]
  %279 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %280 = Transpose(%279, const(i32[2] : {1,0})): // i64[2,2]
  %281 = Conv2D(PadMode.Constant, %278, const(f32[48,144,1,1]), const(f32[48]), const(i64[2] : {1L,1L}), %280, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,48,?,?]
  %282 = Binary(BinaryOp.Add, %281, %238): // f32[?,?,?,?]
  %283 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %284 = Transpose(%283, const(i32[2] : {1,0})): // i64[2,2]
  %285 = Conv2D(PadMode.Constant, %282, const(f32[288,48,1,1]), const(f32[288]), const(i64[2] : {1L,1L}), %284, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,288,?,?]
  %286 = HardSigmoid(%285, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,288,?,?]
  %287 = Binary(BinaryOp.Mul, %285, %286): // f32[?,?,?,?]
  %288 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %289 = Transpose(%288, const(i32[2] : {1,0})): // i64[2,2]
  %290 = Conv2D(PadMode.Constant, %287, const(f32[288,1,5,5]), const(f32[288]), const(i64[2] : {2L,2L}), %289, const(i64[2] : {1L,1L}), const(i64 : 288), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,288,?,?]
  %291 = HardSigmoid(%290, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,288,?,?]
  %292 = Binary(BinaryOp.Mul, %290, %291): // f32[?,?,?,?]
  %293 = ShapeOf(%292): // i64[4]
  %294 = (const(i32 : 2)): // (i32)
  
  %295 = Stack(%294, const(i32 : 0)): // i32[1]
  %296 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %297 = (%296): // (i32)
  
  %298 = Stack(%297, const(i32 : 0)): // i32[1]
  %299 = Slice(%293, %295, %298, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %300 = Squeeze(%299, const(i64[1] : {0L})): // i64
  %301 = ShapeOf(%292): // i64[4]
  %302 = (const(i32 : 3)): // (i32)
  
  %303 = Stack(%302, const(i32 : 0)): // i32[1]
  %304 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %305 = (%304): // (i32)
  
  %306 = Stack(%305, const(i32 : 0)): // i32[1]
  %307 = Slice(%301, %303, %306, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %308 = Squeeze(%307, const(i64[1] : {0L})): // i64
  %309 = (%300, %308): // (i64, i64)
  
  %310 = Stack(%309, const(i32 : 0)): // i64[2]
  %311 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %312 = Transpose(%311, const(i32[2] : {1,0})): // i64[2,2]
  %313 = ReduceWindow2D(ReduceOp.Mean, %292, const(f32 : 0), %310, const(i64[2] : {1L,1L}), %312, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %314 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %315 = Transpose(%314, const(i32[2] : {1,0})): // i64[2,2]
  %316 = Conv2D(PadMode.Constant, %313, const(f32[72,288,1,1]), const(f32[72]), const(i64[2] : {1L,1L}), %315, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,72,?,?]
  %317 = Relu(%316): // f32[?,72,?,?]
  %318 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %319 = Transpose(%318, const(i32[2] : {1,0})): // i64[2,2]
  %320 = Conv2D(PadMode.Constant, %317, const(f32[288,72,1,1]), const(f32[288]), const(i64[2] : {1L,1L}), %319, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,288,?,?]
  %321 = HardSigmoid(%320, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,288,?,?]
  %322 = Binary(BinaryOp.Mul, %321, %292): // f32[?,?,?,?]
  %323 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %324 = Transpose(%323, const(i32[2] : {1,0})): // i64[2,2]
  %325 = Conv2D(PadMode.Constant, %322, const(f32[96,288,1,1]), const(f32[96]), const(i64[2] : {1L,1L}), %324, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,96,?,?]
  %326 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %327 = Transpose(%326, const(i32[2] : {1,0})): // i64[2,2]
  %328 = Conv2D(PadMode.Constant, %325, const(f32[576,96,1,1]), const(f32[576]), const(i64[2] : {1L,1L}), %327, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %329 = HardSigmoid(%328, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %330 = Binary(BinaryOp.Mul, %328, %329): // f32[?,?,?,?]
  %331 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %332 = Transpose(%331, const(i32[2] : {1,0})): // i64[2,2]
  %333 = Conv2D(PadMode.Constant, %330, const(f32[576,1,5,5]), const(f32[576]), const(i64[2] : {1L,1L}), %332, const(i64[2] : {1L,1L}), const(i64 : 576), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %334 = HardSigmoid(%333, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %335 = Binary(BinaryOp.Mul, %333, %334): // f32[?,?,?,?]
  %336 = ShapeOf(%335): // i64[4]
  %337 = (const(i32 : 2)): // (i32)
  
  %338 = Stack(%337, const(i32 : 0)): // i32[1]
  %339 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %340 = (%339): // (i32)
  
  %341 = Stack(%340, const(i32 : 0)): // i32[1]
  %342 = Slice(%336, %338, %341, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %343 = Squeeze(%342, const(i64[1] : {0L})): // i64
  %344 = ShapeOf(%335): // i64[4]
  %345 = (const(i32 : 3)): // (i32)
  
  %346 = Stack(%345, const(i32 : 0)): // i32[1]
  %347 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %348 = (%347): // (i32)
  
  %349 = Stack(%348, const(i32 : 0)): // i32[1]
  %350 = Slice(%344, %346, %349, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %351 = Squeeze(%350, const(i64[1] : {0L})): // i64
  %352 = (%343, %351): // (i64, i64)
  
  %353 = Stack(%352, const(i32 : 0)): // i64[2]
  %354 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %355 = Transpose(%354, const(i32[2] : {1,0})): // i64[2,2]
  %356 = ReduceWindow2D(ReduceOp.Mean, %335, const(f32 : 0), %353, const(i64[2] : {1L,1L}), %355, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %357 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %358 = Transpose(%357, const(i32[2] : {1,0})): // i64[2,2]
  %359 = Conv2D(PadMode.Constant, %356, const(f32[144,576,1,1]), const(f32[144]), const(i64[2] : {1L,1L}), %358, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,144,?,?]
  %360 = Relu(%359): // f32[?,144,?,?]
  %361 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %362 = Transpose(%361, const(i32[2] : {1,0})): // i64[2,2]
  %363 = Conv2D(PadMode.Constant, %360, const(f32[576,144,1,1]), const(f32[576]), const(i64[2] : {1L,1L}), %362, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %364 = HardSigmoid(%363, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %365 = Binary(BinaryOp.Mul, %364, %335): // f32[?,?,?,?]
  %366 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %367 = Transpose(%366, const(i32[2] : {1,0})): // i64[2,2]
  %368 = Conv2D(PadMode.Constant, %365, const(f32[96,576,1,1]), const(f32[96]), const(i64[2] : {1L,1L}), %367, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,96,?,?]
  %369 = Binary(BinaryOp.Add, %368, %325): // f32[?,?,?,?]
  %370 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %371 = Transpose(%370, const(i32[2] : {1,0})): // i64[2,2]
  %372 = Conv2D(PadMode.Constant, %369, const(f32[576,96,1,1]), const(f32[576]), const(i64[2] : {1L,1L}), %371, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %373 = HardSigmoid(%372, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %374 = Binary(BinaryOp.Mul, %372, %373): // f32[?,?,?,?]
  %375 = Reshape(const(i64[4] : {2L,2L,2L,2L}), const(i32[2] : {2,-1})): // i64[2,2]
  %376 = Transpose(%375, const(i32[2] : {1,0})): // i64[2,2]
  %377 = Conv2D(PadMode.Constant, %374, const(f32[576,1,5,5]), const(f32[576]), const(i64[2] : {1L,1L}), %376, const(i64[2] : {1L,1L}), const(i64 : 576), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %378 = HardSigmoid(%377, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %379 = Binary(BinaryOp.Mul, %377, %378): // f32[?,?,?,?]
  %380 = ShapeOf(%379): // i64[4]
  %381 = (const(i32 : 2)): // (i32)
  
  %382 = Stack(%381, const(i32 : 0)): // i32[1]
  %383 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %384 = (%383): // (i32)
  
  %385 = Stack(%384, const(i32 : 0)): // i32[1]
  %386 = Slice(%380, %382, %385, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %387 = Squeeze(%386, const(i64[1] : {0L})): // i64
  %388 = ShapeOf(%379): // i64[4]
  %389 = (const(i32 : 3)): // (i32)
  
  %390 = Stack(%389, const(i32 : 0)): // i32[1]
  %391 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %392 = (%391): // (i32)
  
  %393 = Stack(%392, const(i32 : 0)): // i32[1]
  %394 = Slice(%388, %390, %393, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %395 = Squeeze(%394, const(i64[1] : {0L})): // i64
  %396 = (%387, %395): // (i64, i64)
  
  %397 = Stack(%396, const(i32 : 0)): // i64[2]
  %398 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %399 = Transpose(%398, const(i32[2] : {1,0})): // i64[2,2]
  %400 = ReduceWindow2D(ReduceOp.Mean, %379, const(f32 : 0), %397, const(i64[2] : {1L,1L}), %399, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %401 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %402 = Transpose(%401, const(i32[2] : {1,0})): // i64[2,2]
  %403 = Conv2D(PadMode.Constant, %400, const(f32[144,576,1,1]), const(f32[144]), const(i64[2] : {1L,1L}), %402, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,144,?,?]
  %404 = Relu(%403): // f32[?,144,?,?]
  %405 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %406 = Transpose(%405, const(i32[2] : {1,0})): // i64[2,2]
  %407 = Conv2D(PadMode.Constant, %404, const(f32[576,144,1,1]), const(f32[576]), const(i64[2] : {1L,1L}), %406, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %408 = HardSigmoid(%407, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %409 = Binary(BinaryOp.Mul, %408, %379): // f32[?,?,?,?]
  %410 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %411 = Transpose(%410, const(i32[2] : {1,0})): // i64[2,2]
  %412 = Conv2D(PadMode.Constant, %409, const(f32[96,576,1,1]), const(f32[96]), const(i64[2] : {1L,1L}), %411, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,96,?,?]
  %413 = Binary(BinaryOp.Add, %412, %369): // f32[?,?,?,?]
  %414 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %415 = Transpose(%414, const(i32[2] : {1,0})): // i64[2,2]
  %416 = Conv2D(PadMode.Constant, %413, const(f32[576,96,1,1]), const(f32[576]), const(i64[2] : {1L,1L}), %415, const(i64[2] : {1L,1L}), const(i64 : 1), const(f32[2] : {-Infinityf,Infinityf})): // f32[?,576,?,?]
  %417 = HardSigmoid(%416, const(f32 : 0.16666667), const(f32 : 0.5)): // f32[?,576,?,?]
  %418 = Binary(BinaryOp.Mul, %416, %417): // f32[?,?,?,?]
  %419 = ShapeOf(%418): // i64[4]
  %420 = (const(i32 : 2)): // (i32)
  
  %421 = Stack(%420, const(i32 : 0)): // i32[1]
  %422 = Binary(BinaryOp.Add, const(i32 : 2), const(i32 : 1)): // i32
  %423 = (%422): // (i32)
  
  %424 = Stack(%423, const(i32 : 0)): // i32[1]
  %425 = Slice(%419, %421, %424, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %426 = Squeeze(%425, const(i64[1] : {0L})): // i64
  %427 = ShapeOf(%418): // i64[4]
  %428 = (const(i32 : 3)): // (i32)
  
  %429 = Stack(%428, const(i32 : 0)): // i32[1]
  %430 = Binary(BinaryOp.Add, const(i32 : 3), const(i32 : 1)): // i32
  %431 = (%430): // (i32)
  
  %432 = Stack(%431, const(i32 : 0)): // i32[1]
  %433 = Slice(%427, %429, %432, const(i32[1] : {0}), const(i32[1] : {1})): // i64[?]
  %434 = Squeeze(%433, const(i64[1] : {0L})): // i64
  %435 = (%426, %434): // (i64, i64)
  
  %436 = Stack(%435, const(i32 : 0)): // i64[2]
  %437 = Reshape(const(i64[4] : {0L,0L,0L,0L}), const(i32[2] : {2,-1})): // i64[2,2]
  %438 = Transpose(%437, const(i32[2] : {1,0})): // i64[2,2]
  %439 = ReduceWindow2D(ReduceOp.Mean, %418, const(f32 : 0), %436, const(i64[2] : {1L,1L}), %438, const(i64[2] : {1L,1L}), const(bool : False), const(bool : False)): // f32[?,?,?,?]
  %440 = ShapeOf(%439): // i64[4]
  %441 = GetItem(%440, const(i32 : 0)): // i64
  %442 = Cast(DataTypes.Int64, CastMode.KDefault, const(i32 : 1)): // i64
  %443 = Reduce(ReduceOp.Prod, %440, const(i64[1] : {0L}), %442, const(bool : False)): // i64
  %444 = Binary(BinaryOp.Mul, const(i64 : 1), %441): // i64
  %445 = Binary(BinaryOp.Div, %443, %444): // i64
  %446 = (%441, %445): // (i64, i64)
  
  %447 = Stack(%446, const(i32 : 0)): // i64[2]
  %448 = Reshape(%439, %447): // f32[?,?]
  %449 = Transpose(const(f32[512,576]), const(i32[2] : {1,0})): // f32[576,512]
  %450 = MatMul(%448, %449): // f32[?,512]
  %451 = Binary(BinaryOp.Add, %450, const(f32[512])): // f32[?,?]
  %452 = BatchNormalization(%451, const(f32[512]), const(f32[512]), const(f32[512]), const(f32[512]), const(f32 : 1E-05), const(f32 : 0.9)): // f32[?,?]
  %453 = Relu(%452): // f32[?,?]
  %454 = Transpose(const(f32[256,512]), const(i32[2] : {1,0})): // f32[512,256]
  %455 = MatMul(%453, %454): // f32[?,256]
  %456 = Binary(BinaryOp.Add, %455, const(f32[256])): // f32[?,?]
  %457 = BatchNormalization(%456, const(f32[256]), const(f32[256]), const(f32[256]), const(f32[256]), const(f32 : 1E-05), const(f32 : 0.9)): // f32[?,?]
  %458 = Relu(%457): // f32[?,?]
  %459 = Transpose(const(f32[6,256]), const(i32[2] : {1,0})): // f32[256,6]
  %460 = MatMul(%458, %459): // f32[?,6]
  %461 = Binary(BinaryOp.Add, %460, const(f32[6] : {0.0014595619f,0.001424954f,0.012488966f,0.016606119f,-0.0116832135f,-0.021904912f})): // f32[?,?]
}
