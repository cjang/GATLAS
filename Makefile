
#ATI_SDK = /opt/ati-stream-sdk-v2.0-lnx64
ATI_SDK = $(ATISTREAMSDKROOT)
NVIDIA_SDK = /opt/NVIDIA_GPU_Computing_SDK

GNU_CC = /usr/local/bin/gcc
GNU_CFLAGS = -std=c99 -funsafe-math-optimizations -funroll-all-loops

GNU_CXX = /usr/local/bin/g++
GNU_CXXFLAGS = -funsafe-math-optimizations -funroll-all-loops
#-O3 -funroll-all-loops  -fexpensive-optimizations -ffast-math -finline-functions -frerun-loop-opt -static-libgcc

AR=ar
RANLIB=ranlib

ATI_CFLAGS = -I$(ATI_SDK)/include
ATI_OPENCL_LDFLAGS = -L$(ATI_SDK)/lib/x86_64 -lOpenCL
ATI_CAL_LDFLAGS = -L$(ATI_SDK)/lib/x86_64 -laticalrt -laticalcl

NVIDIA_CFLAGS = -I$(NVIDIA_SDK)/OpenCL/common/inc
NVIDIA_LDFLAGS = -L$(NVIDIA_SDK)/C/common/lib/linux -lOpenCL

GATLAS_LDFLAGS = -L. -lgatlas


.cpp.o :
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@


OCL_OBJECT_CODE = \
	OCLUtil.o \
	OCLBase.o \
	OCLApp.o \
	OCLAppUtil.o \

GATLAS_OBJECT_CODE = \
	GatlasBenchmark.o \
	GatlasCodeText.o \
	GatlasFormatting.o \
	GatlasOperator.o \
	GatlasQualifier.o \
	GatlasType.o

KERNEL_OBJECT_CODE = \
	KernelBaseMatmul.o \
	KernelGenMatmulBuffer.o \
	KernelGenMatmulImage.o \
	KernelMatmulBuffer.o \
	KernelMatmulImage.o

LIB_OBJECT_CODE = \
	$(OCL_OBJECT_CODE) \
	$(GATLAS_OBJECT_CODE) \
	$(KERNEL_OBJECT_CODE)

EXECUTABLES = \
	oclInfo \
	pmm_buffer pgemm_buffer pmm_image pgemm_image \
	bmm_buffer bgemm_buffer bmm_image bgemm_image



# default target for all binaries
binaries : libgatlas.a $(EXECUTABLES)



# minimal matrix multiply GPU characterization
benchdata :
	mkdir benchdata

benchgemmbuffer : benchdata
	@echo benchmarking SGEMM using memory buffers
	@echo warning - this may take a few hours (two hours on my HD 5870)
	./benchMatmul gpu benchdata gemm buffer 1024 5632 256
	./benchMatmul gpu benchdata gemm buffer 1600 5600 800

benchgemmimage : benchdata
	@echo benchmarking SGEMM using images
	@echo warning - this may take a few hours (about 90 minutes my HD 5870)
	./benchMatmul gpu benchdata gemm image 1024 5632 256
	./benchMatmul gpu benchdata gemm image 1600 5600 800

benchmmbuffer : benchdata
	@echo benchmarking matrix multiply using memory buffers
	@echo warning - this may take a few hours (two hours on my HD 5870)
	./benchMatmul gpu benchdata mm buffer 1024 5632 256
	./benchMatmul gpu benchdata mm buffer 1600 5600 800

benchmmimage : benchdata
	@echo benchmarking matrix multiply using images
	@echo warning - this may take a few hours (about 90 minutes my HD 5870)
	./benchMatmul gpu benchdata mm image 1024 5632 256
	./benchMatmul gpu benchdata mm image 1600 5600 800

benchgemm : benchgemmbuffer benchgemmimage
benchmm : benchmmbuffer benchmmimage

benchbuffer: benchgemmbuffer benchmmbuffer
benchimage : benchgemmimage benchmmimage

benchall : benchgemmbuffer benchgemmimage benchmmbuffer benchmmimage



# assemble benchmark data in tabular spreadsheet form
chart : benchdata
	./chartMatmul benchdata gemm > chartgemm.tsv
	./chartMatmul benchdata mm > chartmm.tsv



# very detailed GPU characterization with PCIe bus data transfer costs
fullbenchdata :
	mkdir fullbenchdata

fullbenchgemmbuffer : fullbenchdata
	./benchMatmul gpu fullbenchdata gemm buffer 128 5632 64
	./benchMatmul gpu fullbenchdata gemm buffer to 128 5632 64
	./benchMatmul gpu fullbenchdata gemm buffer from 128 5632 64
	./benchMatmul gpu fullbenchdata gemm buffer tofrom 128 5632 64
	./benchMatmul gpu fullbenchdata gemm buffer 100 5600 100
	./benchMatmul gpu fullbenchdata gemm buffer to 100 5600 100
	./benchMatmul gpu fullbenchdata gemm buffer from 100 5600 100
	./benchMatmul gpu fullbenchdata gemm buffer tofrom 100 5600 100

fullbenchgemmimage : fullbenchdata
	./benchMatmul gpu fullbenchdata gemm image 128 5632 64
	./benchMatmul gpu fullbenchdata gemm image to 128 5632 64
	./benchMatmul gpu fullbenchdata gemm image from 128 5632 64
	./benchMatmul gpu fullbenchdata gemm image tofrom 128 5632 64
	./benchMatmul gpu fullbenchdata gemm image 100 5600 100
	./benchMatmul gpu fullbenchdata gemm image to 100 5600 100
	./benchMatmul gpu fullbenchdata gemm image from 100 5600 100
	./benchMatmul gpu fullbenchdata gemm image tofrom 100 5600 100

fullbenchgemm : fullbenchgemmbuffer fullbenchgemmimage



# archive library
libgatlas.a : $(LIB_OBJECT_CODE)
	$(AR) qc $@ $(LIB_OBJECT_CODE)
	$(RANLIB) $@



# OpenCL information utility
oclInfo : oclInfo.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS)



# print matrix multiply
pmm_buffer.o : printMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelMatmulBuffer.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelMatmulBuffer"

pgemm_buffer.o : printMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelGenMatmulBuffer.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelGenMatmulBuffer"

pmm_image.o : printMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelMatmulImage.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelMatmulImage"

pgemm_image.o : printMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelGenMatmulImage.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelGenMatmulImage"

pmm_buffer : pmm_buffer.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS)

pgemm_buffer : pgemm_buffer.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS)

pmm_image : pmm_image.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS)

pgemm_image : pgemm_image.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS)


# benchmark matrix multiply
bmm_buffer.o : benchMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelMatmulBuffer.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelMatmulBuffer"

bgemm_buffer.o : benchMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelGenMatmulBuffer.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelGenMatmulBuffer"

bmm_image.o : benchMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelMatmulImage.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelMatmulImage"

bgemm_image.o : benchMatmul.cpp
	rm -f KernelFile.hpp
	ln -s KernelGenMatmulImage.hpp KernelFile.hpp
	$(GNU_CXX) -c $(GNU_CXXFLAGS) $(ATI_CFLAGS) $< -o $@ -DKERNEL_CLASS="KernelGenMatmulImage"

bmm_buffer : bmm_buffer.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS) -lm

bgemm_buffer : bgemm_buffer.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS) -lm

bmm_image : bmm_image.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS) -lm

bgemm_image : bgemm_image.o libgatlas.a
	$(GNU_CXX) -o $@ $< $(ATI_OPENCL_LDFLAGS) $(GATLAS_LDFLAGS) -lm



# documentation, needs GraphViz
matmulOverview.svg : matmulOverview.dpp
	./dotpp.pl $< | dot -Tsvg > $@



clean :
	rm -f *.o KernelFile.hpp libgatlas.a $(EXECUTABLES)

# do not want to accidentally delete benchmark data with make clean
veryclean : clean
	rm -rf benchdata
	rm -rf fullbenchdata

