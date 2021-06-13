#include <stdio.h>
#include <sys/stat.h>
#define PNG_sRGB_PROFILE_CHECKS 1
#include <png.h>
#include <tensorflow/c/c_api.h>


typedef struct {
  float *data;
  int width;
  int height;
  int channels;
} image_t;

static void
free_tensor(void *data, size_t len, void *arg) {
  free(data);
}

static void
free_buffer(void *data, size_t length) {
  free(data);
}

static TF_Buffer*
load_graph_def(const char *file) {
  FILE *fp = NULL;
  void *data = NULL;
  struct stat st;
  TF_Buffer *buf = NULL;

  fp = fopen(file, "rb");
  if (fp == NULL) goto err;
  if (fstat(fileno(fp), &st) < 0) goto err;
  data = malloc(st.st_size);
  if (data == NULL) goto err;
  if (fread(data, st.st_size, 1, fp) < 1) goto err;
  fclose(fp);
  fp = NULL;

  buf = TF_NewBuffer();
  buf->data = data;
  buf->length = st.st_size;
  buf->data_deallocator = free_buffer;
  return buf;

err:
  if (data) free(data);
  if (fp) free(fp);
  return NULL;
}

static int
run_session(
    TF_Graph* graph,
    TF_Output* input, TF_Tensor** input_tensor,
    TF_Output* output, TF_Tensor** output_tensor) {
  TF_Status* status = NULL;
  TF_Session* sess = NULL;
  TF_SessionOptions* options = NULL;

  status = TF_NewStatus();
  options = TF_NewSessionOptions();
  sess = TF_NewSession(graph, options, status);

  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "could not get session: %s\n", TF_Message(status));
    goto err;
  }

  TF_SessionRun(sess,
      NULL,
      input, input_tensor, 1,
      output, output_tensor, 1,
      NULL, 0,
      NULL,
      status);

  if (TF_GetCode(status) != TF_OK) {
    fprintf (stderr, "could not run session: %s\n", TF_Message (status));
    goto err;
  }

  TF_CloseSession(sess, status);
  if (TF_GetCode (status) != TF_OK) {
    fprintf (stderr, "could not close session: %s\n", TF_Message (status));
  }

  TF_DeleteSession(sess, status);
  if (TF_GetCode (status) != TF_OK) {
    fprintf (stderr, "could not delete session: %s\n", TF_Message (status));
  }
  return 0;

err:
  TF_DeleteSessionOptions(options);
  TF_DeleteStatus(status);
  return 1;
}

static size_t
max_value(TF_Tensor* t) {
  const float* data = TF_TensorData(t);
  int i;

  size_t element_count = 1;
  const int num_dims = TF_NumDims(t);
  for (i=0; i < num_dims; i++) {
    element_count *= TF_Dim(t, i);
  }

  float mx = data[0];
  size_t mi = 0;
  for (i=1; i < element_count; i++) {
    if (data[i] > mx) {
      mx = data[i];
      mi = i;
    }
  }

  return mi;
}

static int
invoke_session(
    TF_Graph* graph,
    const char *input_layer_name,
    const char *output_layer_name,
    const image_t* image) {
  int i;

  int64_t* input_dims = NULL;
  int64_t* output_dims = NULL;
  float* output_tensor_data = NULL;

  TF_Tensor* input_tensor = NULL;
  TF_Tensor* output_tensor = NULL;

  TF_Operation* input_op = TF_GraphOperationByName(graph, input_layer_name);
  if (input_op == NULL) {
    fprintf(stderr, "could not get input operation: %s\n", input_layer_name);
    goto err;
  }

  TF_Operation* output_op = TF_GraphOperationByName(graph, output_layer_name);
  if (output_op == NULL) {
    fprintf(stderr, "could not get output operation: %s\n", output_layer_name);
    goto err;
  }

  TF_Output input;
  input.oper = input_op;
  input.index = 0;
  TF_Output output;
  output.oper = output_op;
  output.index = 0;

  TF_Status* status = NULL;
  int input_num_dims = TF_GraphGetTensorNumDims(graph, input, status);
  input_dims = (int64_t*) malloc(input_num_dims * sizeof(int64_t));
  if (input_dims == NULL) goto err;
  TF_GraphGetTensorShape(graph, input, input_dims, input_num_dims, status);
  size_t input_size = 1;
  for (i = 0; i < input_num_dims; i++) {
    if (input_dims[i] < 0) input_dims[i] = -input_dims[i];
    input_size *= input_dims[i];
  }

  int output_num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  output_dims = (int64_t*) malloc(output_num_dims * sizeof(int64_t));
  if (input_dims == NULL) goto err;
  TF_GraphGetTensorShape(graph, output, output_dims, output_num_dims, status);
  size_t output_size = 1;
  for (i = 0; i < output_num_dims; i++) {
    if (output_dims[i] < 0) output_dims[i] = -output_dims[i];
    output_size *= output_dims[i];
  }

  if (input_dims[1] != image->width || input_dims[2] != image->height || input_dims[3] != image->channels) {
    fprintf(stderr, "dimensions mismatched (%i, %i, %i) : (%i, %i, %i)\n",
        input_dims[1], input_dims[2], input_dims[3],
        image->width, image->height, image->channels);
    goto err;
  }

  size_t size = (int)output_size > 0 ? output_size : -output_size;
  output_tensor_data = (float *) malloc(size * sizeof(float));
  if (output_tensor_data == NULL) goto err;
  memset(output_tensor_data, 0, size * sizeof(float));

  input_tensor = TF_NewTensor(
      TF_FLOAT,
      input_dims,
      input_num_dims,
      image->data,
      input_size * sizeof(float),
      free_tensor,
      (void*)"input");

  output_tensor = TF_NewTensor(
      TF_FLOAT,
      output_dims,
      output_num_dims,
      output_tensor_data,
      output_size * sizeof(float),
      free_tensor,
      (void*)"output");

  if (run_session(
      graph,
      &input,
      &input_tensor,
      &output,
      &output_tensor) != 0) goto err;

  int result = max_value(output_tensor);
  free(input_dims);
  free(output_dims);
  free(output_tensor_data);
  TF_DeleteTensor(input_tensor);
  TF_DeleteTensor(output_tensor);
  return result;

err:
  if (input_dims) free(input_dims);
  if (output_dims) free(output_dims);
  if (output_tensor_data) free(output_tensor_data);
  if (input_tensor) TF_DeleteTensor(input_tensor);
  if (output_tensor) TF_DeleteTensor(output_tensor);
  return -1;
}

static void
_png_error_handler(png_structp png_ptr, png_const_charp msg) {
}

static image_t*
load_png(const char *filename) {
  FILE *fp = NULL;
  png_structp hdr = NULL;
  png_infop info = NULL;
  int width, height, channels;
  png_byte ct;
  png_byte bd;

  float *bufs = NULL;
  unsigned char *bytes = NULL;
  unsigned char **bytes_refs = NULL;
  image_t* image = NULL;

  fp = fopen(filename, "rb");
  if (fp == NULL) goto err;

  hdr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (hdr == NULL) goto err;
  info = png_create_info_struct(hdr);
  if (info == NULL) goto err;

  png_init_io(hdr, fp);
  png_read_info(hdr, info);
  width = png_get_image_width(hdr, info);
  height = png_get_image_height(hdr, info);
  channels = png_get_channels(hdr, info);
  ct = png_get_color_type(hdr, info);
  bd = png_get_bit_depth(hdr, info);
  png_destroy_read_struct(&hdr, &info, NULL);
  fclose(fp);
  fp = NULL;

  if (ct != PNG_COLOR_TYPE_RGB) {
    fprintf(stderr, "invalid color type: %s\n", filename);
    return NULL;
  }

  bytes = (unsigned char *) malloc(width*height*channels*sizeof(unsigned char));
  if (bytes == NULL) goto err;
  bytes_refs = (unsigned char **) malloc(height*sizeof(unsigned char *));
  if (bytes_refs == NULL) goto err;
  for (int i=0; i<height; i++) {
    bytes_refs[i] = &bytes[i * width * channels];
  }

  fp = fopen(filename, "rb");
  if (fp == NULL) goto err;
  hdr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, _png_error_handler, NULL);
  if (hdr == NULL) goto err;
  info = png_create_info_struct(hdr);
  if (info == NULL) goto err;
  png_init_io(hdr, fp);
  png_read_info(hdr, info);
  png_read_image(hdr, bytes_refs);
  png_read_end(hdr, NULL);
  png_destroy_read_struct(&hdr, &info, NULL);
  fclose(fp);
  fp = NULL;

  bufs = (float *) malloc(width * height * channels * sizeof(float));
  if (bufs == NULL) goto err;
  for (int i=0; i < width * height * channels; i++) {
    bufs[i] = (float)bytes[i];
  }

  free(bytes);
  bytes = NULL;
  free(bytes_refs);
  bytes_refs = NULL;

  image = (image_t*) malloc(sizeof(image_t));
  if (image == NULL) goto err;
  image->data = bufs;
  image->width = width;
  image->height = height;
  image->channels = channels;
  return image;

err:
  if (hdr) png_destroy_read_struct(&hdr, &info, NULL);
  if (fp) fclose(fp);
  if (bufs) free(bufs);
  if (bytes) free(bytes);
  if (bytes_refs) free(bytes_refs);
  if (image) free(image);
  return NULL;
}

int
main(int argc, char *argv[]) {
  TF_Buffer *graph_def = NULL;

  graph_def = load_graph_def(argv[1]);
  if (graph_def == NULL) {
    fprintf(stderr, "unable to load graph def\n");
    return 1;
  }

  TF_Graph *graph = TF_NewGraph();
  TF_Status *status = TF_NewStatus();
  TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions(opts);

  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "unable to load graph: %s\n", TF_Message(status));
    goto err;
  }

  image_t* image = load_png(argv[2]);
  if (image == NULL) {
    fprintf(stderr, "unable to load image: %s\n", argv[2]);
    goto err;
  }

  int i, nbytes = image->width * image->height * image->channels;
  for (i = 0; i < nbytes; i++) {
    float v = image->data[i];
    image->data[i] = ((v / 255.0f) - 0.5f) * 2.0f;
  }

  int result = invoke_session(graph, "input", "MobilenetV1/Predictions/Reshape_1", image);
  if (result < 0) goto err;
  printf("result is %i\n", result);

  if (image) free(image);
  if (status) TF_DeleteStatus(status);
  if (graph) TF_DeleteGraph(graph);
  return 0;

err:
  if (image) free(image);
  if (status) TF_DeleteStatus(status);
  if (graph) TF_DeleteGraph(graph);
  if (graph_def) TF_DeleteBuffer(graph_def);

  return 1;
}
