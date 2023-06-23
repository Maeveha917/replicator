use crate::{
    *,
    visualise,
    neural_net::*
};
use nalgebra::{DVector,DMatrix};
use rand::{self,Rng};

//the below two basically just demo memorisation since one hot doesnt convey anything abt the
//number being represented
pub fn one_hot_binary_to_denary(){
    let mut rng = rand::thread_rng();

    let mut test_network = FeedForward::new(Loss::Squared, Activation::LeakyReLU(0.01), 4, 10)
        .with_layer(Activation::LeakyReLU(0.01), 25)
        .with_layer(Activation::LeakyReLU(0.01), 25)
        .with_layer(Activation::LeakyReLU(0.01), 10);

    for _ in 0..50_000{
        let number = rng.gen_range(0..10);
        let mut denary = vec![0.0;10];
        denary[number]= 1.0;

        let denary = DVector::from_vec(denary);

        let binary = DVector::from_vec(vec![
            ((number >> 0) & 1) as f64,
            ((number >> 1) & 1) as f64,
            ((number >> 2) & 1) as f64,
            ((number >> 3) & 1) as f64,
        ]);

        test_network.backprop(binary.clone(), denary, 0.005);
        let result = test_network.evaluate(binary.clone());
        println!("\nnumber:{}\ninput:",number);
        visualise::index_labled_activations(&binary);
        println!("output:");
        visualise::index_labled_activations(&result);
    }
}
pub fn one_hot_denary_to_binary(){
    let mut rng = rand::thread_rng();

    let mut test_network = FeedForward::new(Loss::Squared, Activation::LeakyReLU(0.01), 10, 10)
        .with_layer(Activation::LeakyReLU(0.01), 25)
        .with_layer(Activation::LeakyReLU(0.01), 25)
        .with_layer(Activation::LeakyReLU(0.01), 4);

    for _ in 0..50_000{
        let number = rng.gen_range(0..10);
        let mut denary = vec![0.0;10];
        denary[number]= 1.0;

        let denary = DVector::from_vec(denary);

        let binary = DVector::from_vec(vec![
            ((number >> 0) & 1) as f64,
            ((number >> 1) & 1) as f64,
            ((number >> 2) & 1) as f64,
            ((number >> 3) & 1) as f64,
        ]);

        test_network.backprop(denary.clone(), binary, 0.005);
        let result = test_network.evaluate(denary.clone());
        println!("\nnumber:{}\ninput:",number);
        visualise::index_labled_activations(&denary);
        println!("output:");
        visualise::index_labled_activations(&result);
    }
}

pub fn xor_gate(){
    let mut rng = rand::thread_rng();

    let mut test_network = FeedForward::new(Loss::Squared, Activation::ReLU, 2, 2)
        .with_layer(Activation::ReLU, 4)
        .with_layer(Activation::ReLU, 4)
        .with_layer(Activation::ReLU, 1);

    for i in 0..50_000{
        let number = rng.gen_range(0..4);
        let input = DVector::from_vec(vec![
            ((number >> 0) &1) as f64,
            ((number >> 1) &1) as f64,
        ]);

        let target = DVector::from_fn(1, |_,_|
            (((number >> 0) &1) ^ ((number >> 1) &1)) as f64);

        test_network.backprop(input.clone(), target.clone(), 0.005);

        let result = test_network.evaluate(input.clone());
        println!("\ninput:");
        visualise::index_labled_activations(&input);
        println!("target:");
        visualise::index_labled_activations(&target);
        println!("output:");
        visualise::index_labled_activations(&result);

    }
}

pub fn mnist_gan(){

    let unlabled_array = csv_to_array("/home/dylan/Documents/Puter/TrainingData/MnistNumbers/test.csv").unwrap();

    let mut unlabled_mnist = unlabled_array.iter()
        .map(|x| {
            DVector::from_iterator(x.len(), 
            //turn vec of ints to floats
            x.into_iter()
                .map(|x| (*x as f64)/255.0))
        });
   
    let mut real_scores:Vec<f64> = Vec::new();
    let mut fake_scores:Vec<f64> = Vec::new();

    let positive = DVector::repeat(1, 1.0);
    let negative = DVector::repeat(1, 0.0);

    let generator = FeedForward::new(Loss::Squared, Activation::LeakyReLU(0.001), 64, 64)
        .with_layer(Activation::LeakyReLU(0.001), 64)
        .with_layer(Activation::Sigmoid, 784);

    let discriminator = FeedForward::new(Loss::Squared, Activation::LeakyReLU(0.001), 784, 729)
        .with_layer(Activation::LeakyReLU(0.001), 32)
        .with_layer(Activation::Sigmoid, 1);

    let mut gan = GAN::new(generator, discriminator);

    for i in 0..100{
        println!("Cycle {}", i);
        println!("training discriminator..");
        for e in 0..5{
            println!("example set {}",e);
            for _ in 0..3{
                let mnist = unlabled_mnist.next().unwrap();
                gan.train_discriminator(mnist, positive.clone(), 0.05);
            }
            for _ in 0..3{
                let fake = gan.generate(); 
                gan.train_discriminator(fake, negative.clone(), 0.05);
            }
            let mnist = unlabled_mnist.next().unwrap();
            let score = gan.discriminator.evaluate(mnist)[0];
            println!("  real data result: {}", score);
            real_scores.push(score);

            let fake = gan.generate();
            let score = gan.discriminator.evaluate(fake)[0];
            println!("  fake data result: {}", score);
            fake_scores.push(score);
        }
        println!("training generator..");
        for _ in 0..3{
            gan.train_generator(positive.clone(), 0.05);
        }
        println!("generation example:");
        visualise::grid_activations(&gan.generate(),28);
    }

    //save results 
    let results_vec:Vec<(f64,f64,f64)> = real_scores.into_iter()
        .zip(fake_scores.into_iter())
        .map(|(real,fake)| {
            let real_loss = Loss::Squared.evaluate(1.0-real);
            let fake_loss = Loss::Squared.evaluate(0.0-fake);
            (real,fake,(real_loss+fake_loss)*0.5)
        })
        .collect();

    println!("saving results.. {:?}",array_to_csv(results_vec));

    //save image
    save_grayscale(gan.generate(),"output_a.jpg", 28, 28);
    save_grayscale(gan.generate(),"output_b.jpg", 28, 28);
    save_grayscale(gan.generate(),"output_c.jpg", 28, 28);
}
