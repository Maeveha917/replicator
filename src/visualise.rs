use nalgebra::{DVector,DMatrix};

pub fn act_to_strength(value:f64)->char{
    //number put into range of 0-1
    if value < 0.2{
        ' '
    }else if value < 0.4{
        '░'
    }else if value < 0.6{
        '▒'
    }else if value < 0.8{
        '▓'
    }else {
        '█'
    }
}

pub fn index_labled_activations(activations:&DVector<f64>){
   println!("index|shaded|value");
   for (act,index) in activations.iter().zip(0..activations.len()){
       println!("{}|{}|{}",index,act_to_strength(*act),act);
   } 
}

pub fn grid_activations(activations:&DVector<f64>,width:usize){
    for i in 0..activations.len(){
        if i%width == 0{ println!()};
        print!("{}",act_to_strength(activations[i]));
    }
}
