use nalgebra::{DVector,DMatrix};
use std::cmp::Ordering;
use rand::{self, Rng, rngs::ThreadRng};

pub struct network_stats{
    pub loss: f64,
}

#[derive(Clone,PartialEq,Debug)]
//activation functions for the network
pub enum Activation{
    ReLU,
    LeakyReLU(f64),
    Sigmoid,
    Tanh
}
impl Activation {
    //applies whichever act\ffivation function selected to input
    pub fn evaluate(&self,input:f64)->f64{
        match self {
            Self::ReLU=> if input>0.0 {input} else {0.0},
            Self::LeakyReLU(negative_gradient)=> if input >0.0 {input} else {input*negative_gradient},
            Self::Sigmoid=>1.0/(1.0+f64::exp(-input)),
            Self::Tanh=>f64::tanh(input)
        }
    }
    //applies derivative of activation func to input
    pub fn evaluate_derived(&self,input:f64)->f64{
        match self{
            Self::ReLU=> if input>0.0{1.0} else{0.0},
            Self::LeakyReLU(negative_gradient)=> if input>0.0{1.0} else{*negative_gradient},
            Self::Sigmoid=>self.evaluate(input)*(1.0-self.evaluate(input)),
            Self::Tanh =>1.0-f64::tanh(input).powi(2)
        }
    } 
}

#[derive(Clone,PartialEq,Debug)]
//function used to evaluate model preformance
pub enum Loss{
    Absolute,
    Squared,
}
impl Loss{
    pub fn evaluate(&self,difference:f64)->f64{
        match self{
            Self::Absolute=> difference,
            Self::Squared=> difference*difference,
        }
    }
    pub fn evaluate_derived(&self,difference:f64)->f64{
        match self{
            Self::Absolute=> match difference.total_cmp(&0.0){
                Ordering::Greater=>1.0,
                Ordering::Equal=>0.0,
                Ordering::Less=>-1.0
            },
            Self::Squared=>2.0*difference,
        }
    }
}

//gan implimented lazily with a feedforward
pub struct GAN{
    rng: ThreadRng,
    pub generator: FeedForward,
    pub discriminator: FeedForward
}
impl GAN{
    pub fn new(generator:FeedForward,discriminator:FeedForward)->GAN{
        GAN { 
            rng: rand::thread_rng(), 
            generator, 
            discriminator,
        }
    }
    pub fn generate(&mut self)->DVector<f64>{
        let random_input = DVector::from_fn(self.generator.layers[0].weights.ncols(),
            |_,_| self.rng.gen::<f64>());
        self.generator.evaluate(random_input) 
    }
    pub fn train_discriminator(&mut self,input:DVector<f64>,target:DVector<f64>,step_size:f64){
        self.discriminator.backprop(input, target, step_size);
    }
    pub fn train_generator(&mut self,target:DVector<f64>,step_size:f64){
        let mut input = DVector::from_fn(self.generator.layers[0].weights.ncols(),
            |_,_| self.rng.gen::<f64>());
        let dloss_to_layers = self.dloss_to_layers(input, target);
        for i in 0..self.generator.layers.len(){
            self.generator.layers[i].weights -= dloss_to_layers[i].0.clone() * step_size;
            self.generator.layers[i].biases -= dloss_to_layers[i].1.clone() * step_size;
        }
    }
    //gets loss derivatives with respect to each layers weights and biases given an input 
    pub fn dloss_to_layers(&self,mut input:DVector<f64>,target:DVector<f64>)->Vec<(DMatrix<f64>,DVector<f64>)>{
        //store all layers derivatives
        let mut layers_dact = Vec::with_capacity(self.generator.layers.len()+self.discriminator.layers.len());
        //take input, pass through one layer, then pass result into next until output
        for layer in self.generator.layers.iter(){
            let(layer_out,layer_derivatives) = layer.evaluate_with_derivatives(input);
            //store layers derivatives
            layers_dact.push(layer_derivatives);
            //pass layer output to next layer 
            input = layer_out
        }
        for layer in self.discriminator.layers.iter(){
            let(layer_out,layer_derivatives) = layer.evaluate_with_derivatives(input);
            //store layers derivatives
            layers_dact.push(layer_derivatives);
            //pass layer output to next layer 
            input = layer_out
        }
        //once derivatives of each layers activations are found, use them to find derivative of
        //loss with respect to weights and biases of layer
        //get derivative of loss with respect to network output
        let error = input.clone()-target;
        let dloss_to_output = error.clone().apply_into(|x| *x = self.discriminator.loss.evaluate_derived(*x));
        //store matrix keeping track of chain of derivatives from the end of the network to current
        //layer
        let mut matrix_chain:DMatrix<f64> = DMatrix::identity(dloss_to_output.len(),dloss_to_output.len());
        //initially multiply matrix chain by loss NOTE TO SELF, MAYBE DONT DO THIS
        for (mut row,dloss) in matrix_chain.row_iter_mut().zip(dloss_to_output.iter()){
            row*= *dloss;
        }
        //also store the derivative of the loss with respect to each layers weights and biases
        let mut dloss_layers = Vec::with_capacity(layers_dact.len());
        for layer_derivs in layers_dact.into_iter().rev(){
            //in order to find weights effects on loss, chain matrix's rows are summed to get total
            //effect on loss that each layer activation has
            let dloss_layer_acts = matrix_chain.clone().row_sum();
            let mut dloss_to_weights = layer_derivs.weights.clone();
            let mut dloss_to_biases = layer_derivs.biases.clone();
            //multiply each item corresponding item by one in dloss layer acts 
            for (dloss_layer_acts,(mut weights,mut bias)) in dloss_layer_acts.into_iter()
                .zip(dloss_to_weights.row_iter_mut().zip(dloss_to_biases.iter_mut())){
                weights *= *dloss_layer_acts;
                *bias = *dloss_layer_acts;
            }

            dloss_layers.push((dloss_to_weights,dloss_to_biases));

            matrix_chain *= layer_derivs.input;
        }
        gradient_clip(dloss_layers.into_iter().rev().collect(),1.0)
    }
}

//simple feed forward neural network
pub struct FeedForward{
    //function used for evaluating model preformance
    pub loss: Loss,
    //all layers in the feed forward network
    pub layers: Vec<Layer> 
}
impl FeedForward{
    pub fn from_layer_list(loss:Loss,mut inputs:usize,layers:Vec<(usize,Activation)>)->FeedForward{
        let layers = layers.into_iter()
            .map(|(layer_size,act)|{
                let layer = Layer::new(act, inputs, layer_size);
                inputs = layer_size;
                layer
            }).collect();
        FeedForward{
            loss,
            layers
        }
    }
    //creates new network with just one layer at start
    pub fn new(loss:Loss,activation:Activation,input_count:usize,output_count:usize)->FeedForward{
        let first_layer = Layer::new(activation,input_count,output_count);
        FeedForward{
            loss,
            layers: vec![first_layer]
        }
    }
    //adds another layer to network
    pub fn with_layer(mut self, activation:Activation, layer_size:usize)->FeedForward{
        //find amount of inputs layer will recieve
        let layer_inputs = self.layers.last().unwrap().biases.len();
        //create new layer
        let new_layer = Layer::new(activation,layer_inputs,layer_size);
        //add new layer
        self.layers.push(new_layer);
        self
    }
    //produces output from network
    pub fn evaluate(&self,mut input:DVector<f64>)->DVector<f64>{
        //take input, pass through one layer, then pass result into next until output
        for layer in self.layers.iter(){
            input = layer.evaluate(input);
        }
        input
    }

    //gets loss derivatives with respect to each layers weights and biases given an input 
    pub fn dloss_to_layers(&self,mut input:DVector<f64>,target:DVector<f64>)->Vec<(DMatrix<f64>,DVector<f64>)>{
        //store all layers derivatives
        let mut layers_dact = Vec::with_capacity(self.layers.len());
        //take input, pass through one layer, then pass result into next until output
        for layer in self.layers.iter(){
            let(layer_out,layer_derivatives) = layer.evaluate_with_derivatives(input);
            //store layers derivatives
            layers_dact.push(layer_derivatives);
            //pass layer output to next layer 
            input = layer_out
        }
        //once derivatives of each layers activations are found, use them to find derivative of
        //loss with respect to weights and biases of layer
        //get derivative of loss with respect to network output
        let error = input.clone()-target;
        let dloss_to_output = error.clone().apply_into(|x| *x = self.loss.evaluate_derived(*x));
        //store matrix keeping track of chain of derivatives from the end of the network to current
        //layer
        let mut matrix_chain:DMatrix<f64> = DMatrix::identity(dloss_to_output.len(),dloss_to_output.len());
        //initially multiply matrix chain by loss NOTE TO SELF, MAYBE DONT DO THIS
        for (mut row,dloss) in matrix_chain.row_iter_mut().zip(dloss_to_output.iter()){
            row*= *dloss;
        }
        //also store the derivative of the loss with respect to each layers weights and biases
        let mut dloss_layers = Vec::with_capacity(self.layers.len());
        for layer_derivs in layers_dact.into_iter().rev(){
            //in order to find weights effects on loss, chain matrix's rows are summed to get total
            //effect on loss that each layer activation has
            let dloss_layer_acts = matrix_chain.clone().row_sum();
            let mut dloss_to_weights = layer_derivs.weights.clone();
            let mut dloss_to_biases = layer_derivs.biases.clone();
            //multiply each item corresponding item by one in dloss layer acts 
            for (dloss_layer_acts,(mut weights,mut bias)) in dloss_layer_acts.into_iter()
                .zip(dloss_to_weights.row_iter_mut().zip(dloss_to_biases.iter_mut())){
                weights *= *dloss_layer_acts;
                *bias = *dloss_layer_acts;
            }

            dloss_layers.push((dloss_to_weights,dloss_to_biases));

            matrix_chain *= layer_derivs.input;
        }
        gradient_clip(dloss_layers.into_iter().rev().collect(),1.0)
    }

    pub fn backprop(&mut self,mut input:DVector<f64>,target:DVector<f64>,step_size:f64){
        //get all layer derivatives, applies them to weights and biases
        let dloss_layers = self.dloss_to_layers(input, target);
        //reverse derivatives as they were calculated in backwards order, god i wonder why they did
        //that surely there is no backwards in backpropigation i have not heard of such things
        //println!("\noutput\n{}\nerror\n{}\n\ndloss with respect to network activation\n{}\n",input,error,dloss_to_output);
        //now that derivatives found, decend
        for (mut layer, derivatives) in self.layers.iter_mut().zip(dloss_layers.into_iter()){
            layer.weights -= derivatives.0*step_size;
            layer.biases -= derivatives.1*step_size;
        }
    }
}

pub fn gradient_clip(mut dloss_to_layers:Vec<(DMatrix<f64>,DVector<f64>)>,threshold:f64)->Vec<(DMatrix<f64>,DVector<f64>)>{
    for (weights,biases) in dloss_to_layers.iter_mut(){
        if weights.magnitude() > threshold{
            weights.normalize_mut();
        }
        if biases.magnitude() > threshold{
            biases.normalize_mut();
        }
    }
    dloss_to_layers
}

//stores all relavent derivatives of a layers activations
pub struct LayerDerivatives{
    pub input:DMatrix<f64>,
    pub weights:DMatrix<f64>,
    pub biases:DVector<f64>
}

//layer of neural network
pub struct Layer{
    //activation function used in network
    pub activation: Activation,
    //all layers weights
    pub weights:DMatrix<f64>,
    //all layers biases
    pub biases:DVector<f64>
}
impl Layer{
    //creates new layer with each weight in range -1 to 1 and all biases at zero
    pub fn new(activation:Activation, input_size:usize,layer_size:usize)->Layer{
        let mut rng = rand::thread_rng();
        let weights = DMatrix::from_fn(layer_size,input_size,|_,_| rng.gen_range(-1.0..1.0));
        let biases = DVector::repeat(layer_size, 0.0);
        Layer{
            activation,
            weights,
            biases
        }
    }
    //creates new layer uses defined functions to initialize each weight and bias
    pub fn from_fn<F>(activation:Activation, input_size:usize,layer_size:usize, weight_func:&mut F,bias_func:&mut F)->Layer
    where F : FnMut(usize,usize)->f64{
        let weights = DMatrix::from_fn(layer_size,input_size, weight_func);
        let biases = DVector::from_fn(layer_size, bias_func);
        Layer{
            activation,
            weights,
            biases
        }
    }
    //uses existing weights and biases to create new layer
    pub fn from_existing(activation:Activation, weights:DMatrix<f64>,biases:DVector<f64>)->Layer{
        Layer{
            activation,
            weights,
            biases
        }
    }
    //takes input, outputs result
    pub fn evaluate(&self,mut input:DVector<f64>)->DVector<f64>{
        input /= input.max();
        //get output of layer, pre activation
        let pre_activation = self.weights.clone()*input+self.biases.clone();
        //apply activation function
        pre_activation.apply_into(|x| *x = self.activation.evaluate(*x))
    }
    //gets derivative of layers output with respect to layer input aswell as layer output
    pub fn evaluate_with_derivatives(&self,mut input:DVector<f64>)->(DVector<f64>,LayerDerivatives){
        input /= input.max();
        //get pre activation values in network
        let pre_activation = self.weights.clone()*input.clone()+self.biases.clone();
        //then get post activation
        let post_activation = pre_activation.clone().apply_into(|x| *x = self.activation.evaluate(*x));

        //first get derivative of layer activation with respect to pre activation values
        let dact_to_pre_act = pre_activation.apply_into(|x| *x = self.activation.evaluate_derived(*x));
        
        //use derivative of activation with respect to pre activation values to find the desired
        //derivatives
        let mut respect_to_inputs = self.weights.clone();
        for (mut row,pre_act) in respect_to_inputs.row_iter_mut().zip(dact_to_pre_act.iter()){
            row *= *pre_act;
        }
        let respect_to_weights = dact_to_pre_act.clone()*input.transpose();

        let layer_derivatives = LayerDerivatives{
            input: respect_to_inputs,
            weights:respect_to_weights,
            biases:dact_to_pre_act
        };

        //return output and layer's derivatives
        (post_activation,layer_derivatives)
    }
}
