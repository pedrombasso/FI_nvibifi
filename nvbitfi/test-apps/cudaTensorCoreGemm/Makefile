GPU=1
DEBUG=0
#for radiation setup
LOGS=1
OMP=1

# NVCC optmizations flags
NVCCOPTFLAGS?=

# This is what I use, uncomment if you know your arch and want to specify
# Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]

# # check for cuda 7.0 or lower
# ifneq (, $(findstring cuda-7.0, $(CUDAPATH)))
# else
# # Titan X | Titan Xp
# #ARCH+=     -gencode arch=compute_61,code=[sm_61,compute_61]
# # Tegra X2
# #ARCH+=     -gencode arch=compute_62,code=[sm_62,compute_62]
# # Titan V | Tesla V100
# ARCH+=  -gencode arch=compute_70,code=[sm_70,compute_70]
# # Xavier
# #ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]
# endif

VPATH=./src/
TARGET=cudaTensorCoreGemm
OBJDIR=./obj/

CXX=g++

CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc 
OPTS=-O3
LDFLAGS= 
COMMON= 
CFLAGS= -Wall -Wfatal-errors

RAD_DIR=/home/carol/radiation-benchmarks

INCLUDE=-I$(CUDAPATH)/include -I../common -I/home/carol/radiation-benchmarks/src/include -I../../Common -I$(CUDAPATH)/samples/common/inc -I$(RAD_DIR)/src/cuda/common/



ifeq ($(CUBLAS), 1)
USECUBLAS= --use_cublas
endif

ifeq ($(TENSOR_CORES), 1)
USETENSORS= --tensor_cores
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -DDEBUG
NVCCFLAGS+= -g -G -DDEBUG
endif

ifeq ($(BUILDRELATIVEERROR), 1)
NVCCFLAGS+= -DBUILDRELATIVEERROR=1
endif

CFLAGS+=$(OPTS)
STDVERSION=--std=c++11

COMMON+= $(STDVERSION)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I$(CUDAPATH)/include/
CFLAGS+= -DGPU
LDFLAGS+= -L$(CUDAPATH)/lib64 -lcudart -lcurand -lcublas
endif

ifeq ($(OMP), 1) 
CFLAGS+= -DOMP -fopenmp
endif


OBJ= cudaTensorCoreGemm.o Parameters.o 

ifeq ($(LOGS), 1)
INCLUDE+=-I../../include/ 
NVCCFLAGS+= -DLOGS
CFLAGS+= -DLOGS
LDFLAGS+= -L../../include/ -L$(RAD_DIR)/src/include/ -lLogHelper -DLOGS=1
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
# DEPS = $(wildcard src/*.h) Makefile

NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS)
CFLAGS+= -DNVCCOPTFLAGS="$(strip $(patsubst %-Xptxas,%,$(NVCCOPTFLAGS)))"

all: obj $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(COMMON) $(CFLAGS) $^ $(INCLUDE) -o $@  $(LDFLAGS)  

$(OBJDIR)%.o: %.cpp 
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@ $(INCLUDE) 

$(OBJDIR)%.o: %.cu 
	$(NVCC) -ccbin $(CXX) $(ARCH) $(COMMON) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@ 


obj:
	mkdir -p obj


.PHONY: clean
clean:
	rm -rf $(OBJS) $(TARGET)

            
