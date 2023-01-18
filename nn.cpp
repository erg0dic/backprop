#include <iostream>
#include <cmath>
#include "nn.h"

using namespace std;


template <class T> T sigmoid(T x) {

    T out =  (T)1/((T)1+(T)exp(-x));
    return out;

} 

template <class t> t relu(t x){
    t out = max((t)0, x);
    return out;
}

template <class t> t tanh(t x){

    t out = tanh(x);
    return out;
}

template <class t> t dsigmoid(t x){

    t out = sigmoid<t>(x);
    return out*(1-out);
}

template <class t> t drelu(t x){

    if (x < 0){
        return (t)0;
    }
    else {
        return (t)1;
    }

}

template <class t> t dtanh(t x){
    t c = tanh<t>(x);
    t out = (t)1 - c*c;
    return out;
}


template <class t> t random(t x) {
// get random value in the interval [-x, x]
    t out = (double)x*(2*(double)rand()/(double)RAND_MAX - (double)1 );
    return out;
}

// constructor
template <class t, int h_layers, int in_dim, int out_dim, int h_dim> 
nn::Network<t, h_layers, in_dim, out_dim, h_dim>::Network(t m){

    // init w_in
    for (int i=0; i < in_dim; i++){
        for (int j=0; j < h_dim; j++) {
            w_in[j][i] = random<t>(m);
            // this -> w_in[j][i];
            cout << "w_init: " << w_in[j][i] << " " << this << endl;
        }
    }
    // init w_out
    for (int i=0; i < out_dim; i++){
        for (int j=0; j < h_dim; j++) {
            w_out[i][j] = random<t>(m);
        }
    }
    // init h
    for (int n=0; n < h_layers; n++) {
        for (int i=0; i < h_dim; i++){
            for (int j=0; j < h_dim; j++) {
                w_hiddens[n][i][j] = random<t>(m);
            }
        }
    }
    // init biases /{bias_out}
    for (int i=0; i < h_layers+1; i++){
        for (int j=0; j < h_dim; j++) {
            biases[i][j] = random<t>(m);
        }
    }

    for (int i=0; i<h_dim; i++){
        cout << "biases: " << biases[0][i] << endl;
    }
    // init bias_out
    for (int i=0; i < out_dim; i++){
        bias_out[i] = random<t>(m);
    }


};

template <class t, int h_layers, int in_dim, int out_dim, int h_dim> 
void nn::Network<t, h_layers, in_dim, out_dim, h_dim>::forward(t fin[in_dim]){
    // doing this using relus, but can replace with any other differentiable activation function (c.f. above)

    // store initial input, for use in the backward step
    for (int i = 0; i < in_dim; i++)
    {
        initial[i] = fin[i];
    }
    
    // prop through input layer
    for (int i=0; i<h_dim; i++){
        t store = 0;
        for (int j=0; j<in_dim; j++) {
            store += w_in[i][j]*fin[j];   
        }
        // cout << "store input layer 1: " << store << endl;
        input[i] = relu<t>(store + biases[0][i]);  // apply the activation function
        intermediate_outs[0][i] = input[i];
        //cout << "input layer 1: " << input[i] << endl;
    }

    // prop through hidden layers
    for (int n=0; n < h_layers; n++){
        t aux[h_dim];
        for (int i=0; i<h_dim; i++){
            t store = 0;
            for (int j=0; j<h_dim; j++) {
                store += w_hiddens[n][i][j]*input[j];   
            }
            aux[i] = relu<t>(store + biases[n+1][i]);
            //cout << "hidden layer " << n+2 << ": "<< aux[i] << " untransformed: " << store+biases[n+1][i] << endl;
        }
        // update the inputs
        for (int i=0; i<h_dim; i++){
            input[i] = aux[i];
            intermediate_outs[1+n][i] = input[i];
            //cout << "new input: " << input[i] << endl;
        }
    
    }
    // prop through output layer
    for (int i=0; i<out_dim; i++){
        t store = 0;
        for (int j=0; j<h_dim; j++){
            store += w_out[i][j]*input[j];
        }
        output[i] = store + bias_out[i];
        cout << "output layer: " << output[i] << endl;
    }

}

// probably will need outputs from every single layer to feed into gradient updates
template <class t, int h_layers, int in_dim, int out_dim, int h_dim> 
void nn::Network<t, h_layers, in_dim, out_dim, h_dim>::backward(t true_out[out_dim], t learning_rate){
    
    // compute rms loss
    t rms_loss = 0;
    for (int i=0; i < out_dim; i++) {
        rms_loss += (true_out[i]-output[i])*(true_out[i]-output[i]);
    }
    cout << "loss: " << rms_loss << endl;

    t J_output[out_dim];  // gradient vector w.r.t. the loss
    
    for (int i=0; i < out_dim; i++) {
        J_output[i] = 2*(true_out[i]-output[i]);
        //cout << "J_out: " << J_output[i] << endl;
    }
    
    // compute gradients for the final layer
    t J_wout_hl[out_dim];
    t J_bias_hl[out_dim];

    for (int i = 0; i < out_dim; i++){
        J_wout_hl[i]= output[i]*J_output[i];
        J_bias_hl[i] = J_output[i];
        //cout << "J_out_hl: " << J_wout_hl[i] << endl;
    }

    // compute gradients for the intermediate layers
    t J_w_hiddens[1+h_layers][h_dim];
    t J_bias_hiddens[1+h_layers][h_dim];

    // grads for final hidden layer
    for (int j = 0; j < h_dim; j++){
        t js = 0;
        for (int i = 0; i < out_dim; i++){
            js += J_wout_hl[i]*w_out[i][j]*drelu<t>(intermediate_outs[h_layers][j]);
        }
        J_w_hiddens[h_layers][j] = js;
        J_bias_hiddens[h_layers][j] = js;
    }

    // grads for all layers except final hidden layer
    for (int n = h_layers-1; n >= 0; n--){
        for (int j = 0; j < h_dim; j++){
            t js = 0;
            for (int i = 0; i < h_dim; i++){
                js += J_w_hiddens[n+1][i]*w_hiddens[n+1][i][j]* drelu<t>(intermediate_outs[n][j]);
            }
            J_w_hiddens[n][j] = js;
            J_bias_hiddens[n][j] = js;
        }
    }
 
// *intermediate_outs[h_layers-1][k] postpone to the update step

    // update the weights and biases for the final output layer using the calculated gradients via simple gradient descent
    for (int i = 0; i < out_dim; i++){
        for (int j = 0; j < h_dim; j++){
            w_out[i][j] -= learning_rate*J_wout_hl[i]*intermediate_outs[h_layers][j];
            //cout << "w_out_hl: " << w_out[i][j] << endl;
        }
        bias_out[i] -= learning_rate*J_bias_hl[i];
    }

    // update the weights and biases for the hidden (intermediate) layers
    for (int i = 0; i < h_dim; i++){
        for (int j = 0; j < h_dim; j++){
            w_hiddens[h_layers-1][i][j] -= learning_rate*J_w_hiddens[h_layers][i]*intermediate_outs[h_layers-1][j];
        }
        biases[h_layers][i] -= learning_rate*J_bias_hiddens[h_layers][i];
    }
    // update weights and biases for all except input layer
    for (int n = h_layers-1; n > 0; n--){
        for (int i = 0; i < h_dim; i++){
            for (int j = 0; j < h_dim; j++){
                w_hiddens[n-1][i][j] -= learning_rate*J_w_hiddens[n][i]*intermediate_outs[n-1][j];
            }
            biases[n][i] -= learning_rate*J_bias_hiddens[n][i];
        }
    }

    // update the weights and biases for the input layer
    for (int i = 0; i < h_dim; i++){
        for (int j = 0; j < in_dim; j++){
            w_in[i][j] -= learning_rate*J_w_hiddens[0][i]*initial[j];
        }
        biases[0][i] -= learning_rate*J_bias_hiddens[0][i];
    }

    // DONE!
}

// deconstructor: defaulted in the header file
// template <class t, int h_layers, int in_dim, int out_dim, int h_dim> 
// nn::Network<t, h_layers, in_dim, out_dim, h_dim>::~Network(){

// };

namespace experimental {

// neat idea for quantizing nn arithmetic but not really useful atm...
template <class T>  T ftorial(T x) {

    T out = 1;

    while (x>0) {

        out *= x;
        x--;
    }
    return out;

}

template <class t> t pow(t x, int n){

    t out = 1;
    while (n>0) {
        out *= x;
        n--;
    }
    return out;
}

template <class t> t exp(t x) {
    // arbitray precision exp
    t out = 1;
    for (int i=1; i<16; i++) {
        out += pow<t>(x, i) / ftorial(i);
    }
    return out;
}

}


int main() {

    // debug a few things

    long x = -6, k, m;

    k = sigmoid<long>(x);

    cout << k << endl;
    cout << experimental::ftorial(3) << endl;

    long v = 1;
    cout << experimental::exp<long>(v) << endl;
    cout << (long)exp(v) << endl;
    cout << rand()/(double)RAND_MAX << endl;
    cout << rand()/(double)RAND_MAX << endl;

    cout << "random function test (-2,2): " << random<float>((float)2) << endl;

    srand(2225);
    nn::Network<float, 1,2,1,6> *n = new nn::Network<float, 1,2,1,6>(0.5);
    float inp[4][2] = {{1,1},{1,0},{0,1},{0,0}};
    float out[4][1] = {{0},{1},{1},{0}};
    for (int iterations = 0; iterations < 30000; iterations++){
        for (int j = 0; j < 4; j++)
        {
            cout << j << endl;
            n->forward(inp[j]);
            n->backward(out[j], -0.001);
        }

      
    }

    // // learn an XOR gate
    // float inp[1][8] = {{1,1,1,0,0,1,0,0}};
    // float out[1][4] = {{0,1,1,0}};
    // for (int iterations = 0; iterations < 16; iterations++){
   
    //     n->forward(inp[0]);
    //     n->backward(out[0], -0.2);
      
    // }
    n->forward(inp[0]);
    n->forward(inp[1]);
    n->forward(inp[2]);
    n->forward(inp[3]);

    return 0;

}