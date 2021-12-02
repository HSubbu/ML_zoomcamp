FROM agrigorev/zoomcamp-cats-dogs-lambda:v2
RUN pip install pillow
RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl
#RUN pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
COPY lamda_function.py .

CMD ["lamda_function.lamda_handler"]