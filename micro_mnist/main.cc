/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_mnist/model/mnist_model.h"
#include "tensorflow/lite/micro/examples/micro_mnist/model/mnist_test_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Helper fn to log the shape and datatype of a tensor
void printTensorDetails(TfLiteTensor* tensor,
                        tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Type [%s] Shape :", TfLiteTypeGetName(tensor->type));
  for (int d = 0; d < tensor->dims->size; ++d) {
    error_reporter->Report("%d [ %d]", d, tensor->dims->data[d]);
  }
  error_reporter->Report("");
}


int get_top_prediction(const int32_t* predictions, int num_categories, tflite::ErrorReporter* error_reporter) {
  int max_score = predictions[0];
  int guess = 0;

  for (int category_index = 1; category_index < num_categories;
       category_index++) {
    const int32_t category_score = predictions[category_index];
    if (category_score > max_score) {
      max_score = category_score;
      guess = category_index;
        
    }
  }

  return guess;
}


int main(int argc, char* argv[]) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  error_reporter->Report(
      "Parsing MNIST classifier model FlatBuffer, size %d bytes.",
      mnist_model_tflite_len);
  const tflite::Model* model = ::tflite::GetModel(mnist_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // This pulls in all the operation implementations we need.
  //tflite::ops::micro::AllOpsResolver resolver;
    
  static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
    tflite::ops::micro::Register_RESHAPE());
    micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
    tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
    tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
   
  // Create an area of memory to use for input, output, & intermediate arrays.
  // The size of this will depend on the model you're using, currently
  // determined by experimentation.
  const int tensor_arena_size = 110 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with.
    
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();
  
      // Get information about the models input and output tensors.
  TfLiteTensor* model_input = interpreter.input(0);
  error_reporter->Report("Details of input tensor:");
  printTensorDetails(model_input, error_reporter);
  TfLiteTensor* model_output = interpreter.output(0);
  error_reporter->Report("Details of output tensor:");
  printTensorDetails(model_output, error_reporter);
  
    
  error_reporter->Report("Test model over %d images.\n", mnistSampleCount);

  // perform inference on each test sample and evalute accuracy of model
  int accurateCount = 0;
  const int inputTensorSize = 28 * 28;
  for (int s = 0; s < mnistSampleCount; ++s) {
    // Set value of input tensor
    for (int d = 0; d < inputTensorSize; ++d) {
      model_input->data.f[d] = mnistInput[s][d]/255;
    }

    // perform inference
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed.\n");
      return 1;
    }
      int guess =  get_top_prediction(model_output->data.i32, 10, error_reporter);
    
    error_reporter->Report("Model estimate [%d] training label [%d]",
                          guess, mnistOutput[s]);

    if (guess == mnistOutput[s]) ++accurateCount;
  }

  error_reporter->Report("Test set accuracy was %d percent\n",
                         ((accurateCount * 100) / mnistSampleCount));

  error_reporter->Report("MNIST classifier example completed successfully.\n");

  return 0;
}
