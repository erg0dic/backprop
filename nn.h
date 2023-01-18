#ifndef nn
#define nn 
#include <cstdlib>    // include the c standard library



template <class T, int h_layers, int in_dim, int out_dim, int h_dim> 

class Network {


    private: 
        T w_in[h_dim][in_dim];   // rank of w_in is in_dim i.e. input size or the dim of the nested array
        T w_hiddens[h_layers][h_dim][h_dim];  // plural: have "h_layers" hidden weights (square mats)
        T w_out[out_dim][h_dim];
        T biases[1+h_layers][h_dim]; // biases for each layer excluding the final layer (N.B. "INCLUDING INPUT LAYER")
        T bias_out[out_dim];   // bias for the output layer has different dim
        T input[h_dim];
        T output[out_dim];
        T intermediate_outs[1+h_layers][h_dim];  // intermediate layer outputs for gradient flow
        T initial[in_dim];



    public:
        Network(T);  // construct: initialize network with [-m,m] to be chosen m weights
        ~Network() = default;    // once done, deconstruct: don't leak memory
        void forward(T*);  // standard way of passing a vector/arraay as a parameter! T* <=> T[any_size]
        void backward(T*, T);

};
#endif 