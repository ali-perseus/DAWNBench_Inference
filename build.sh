
MINICONDA_PATH="/path/to/miniconda3"

INCLUDE_PATH=" -I ${MINICONDA_PATH}/include -I ${MINICONDA_PATH}/lib/python3.6/site-packages/hgai/include"
LIB_PATH="-L ${MINICONDA_PATH}/lib/python3.6/site-packages/hgai/lib -L ${MINICONDA_PATH}/lib"
RPATH_OPTION="-Wl,-rpath,${MINICONDA_PATH}/lib:${MINICONDA_PATH}/lib/python3.6/site-packages/hgai/lib"


g++ -O3 -std=c++11 main.cpp ${INCLUDE_PATH} ${LIB_PATH} -I ./  -L ./  -lperseus_inference -lopencv_imgproc -lopencv_core -lopencv_imgcodecs ${RPATH_OPTION}  -o inference_test