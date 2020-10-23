from os import environ

benchmark = environ['BENCHMARK']
NVBITFI_HOME = environ['NVBITFI_HOME']
THRESHOLD_JOBS = int(environ['FAULTS'])

all_apps = {
 
        'simple-faster-rcnn-pytorch': [
            NVBITFI_HOME + '/test-apps/simple-faster-rcnn-pytorch', # workload directory
            'simple-faster-rcnn-pytorch', # binary name
            NVBITFI_HOME + '/test-apps/simple-faster-rcnn-pytorch/', # path to the binary file
            1, # expected runtime
            "" # additional parameters to the run.sh
        ],

        
        'pytorch_vision_resnet': [
            NVBITFI_HOME + '/test-apps/pytorch_vision_resnet', # workload directory
            'pytorch_vision_resnet', # binary name
            NVBITFI_HOME + '/test-apps/pytorch_vision_resnet/', # path to the binary file
            1, # expected runtime
            "" # additional parameters to the run.sh
        ],

        'cudaTensorCoreGemm': [
            NVBITFI_HOME + '/test-apps/cudaTensorCoreGemm', # workload directory
            'cudaTensorCoreGemm', # binary name
            NVBITFI_HOME + '/test-apps/cudaTensorCoreGemm/', # path to the binary file
            1, # expected runtime
            "" # additional parameters to the run.sh
        ],




}

apps = {benchmark : all_apps[benchmark]}
