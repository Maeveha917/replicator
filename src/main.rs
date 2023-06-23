use nalgebra::{DVector,DMatrix};
use rand::prelude::*;
use csv::{Reader,Writer};
use eframe::egui;
use image::RgbImage;

use menus::dvector_to_image;
use neural_net::*;

mod neural_net;
mod demos;
mod menus;
mod visualise;

fn main()-> Result<(),eframe::Error> {
    std::env::set_var("RUST_BACKTRACE", "1");
    let options = eframe::NativeOptions {
        ..Default::default()
    };
    
    eframe::run_native(
        "Replicator",
        options,
        Box::new(|_cc| Box::<GenGui>::default()),
    )
}

#[derive(Debug)]
pub enum CycleState{
    Real,
    Fake,
    Gen
}

#[derive(Clone)]
pub struct SetupData{
    pub generator_layers: Vec<(usize,Activation)>,
    pub discriminator_layers: Vec<(usize,Activation)>,

    pub loss: Loss,

    pub generator_inputs: usize,
    pub images_dir_path: String,
    pub image_dimensions: (usize,usize),
}
impl SetupData{
    pub fn to_running(&mut self)->RunningData{
        let gen = FeedForward::from_layer_list(self.loss.clone(), self.generator_inputs, self.generator_layers.clone());
        let disc = FeedForward::from_layer_list(self.loss.clone(), self.generator_layers.last().unwrap().0,self.discriminator_layers.clone());

        let pos = DVector::from_row_slice(&[1.0,0.0]);
        let neg = DVector::from_row_slice(&[0.0,1.0]);

        RunningData {
            setup_data: self.clone(),
            gan: GAN::new(gen, disc),
            discrim_pos_result: pos,
            discrim_neg_result: neg,
            image: None,
            image_dimensions: self.image_dimensions,
            cycle_data: CycleData::new(),
            step_size: 0.05
        }
    }
}

pub struct RunningData{
    pub setup_data: SetupData,
    pub gan: GAN,

    pub discrim_pos_result:DVector<f64>,
    pub discrim_neg_result:DVector<f64>,

    pub image: Option<egui::ColorImage>,  
    pub image_dimensions: (usize,usize),

    pub cycle_data: CycleData,

    pub step_size:f64
}

pub struct CycleData{
    //training perameters
    pub training: bool,
    pub cycles: usize,
    pub discrim_real_period: usize,
    pub discrim_fake_period: usize,
    pub generator_period: usize,

    //current progress through training
    pub current_cycle: usize,
    pub cycle_state: CycleState,
    pub state_progress: usize
}
impl CycleData{
    pub fn new()->CycleData{
        CycleData {
            training: false,
            cycles: 100, 
            discrim_real_period: 1, 
            discrim_fake_period: 1, 
            generator_period: 1, 
            current_cycle: 1, 
            cycle_state: CycleState::Real, 
            state_progress: 1 
        }
    }
    pub fn advance(&mut self){
        if self.current_cycle>=self.cycles{
            self.training = false;
            return;
        }
        //advance
        self.state_progress -=1;
        if self.state_progress == 0{
            //correct to reflect advancement
            self.cycle_state = match self.cycle_state{
                CycleState::Real=>{
                    self.state_progress = self.discrim_fake_period;
                    CycleState::Fake
                },
                CycleState::Fake=>{
                    self.state_progress = self.generator_period;
                    CycleState::Gen
                },
                CycleState::Gen=>{
                    self.state_progress = self.discrim_real_period;
                    self.current_cycle += 1;
                    CycleState::Real
                }
            };
        }
    }
}

enum GuiState{
    Setup(SetupData),
    Running(RunningData)
}

struct GenGui{
    state: GuiState,
}
impl Default for GenGui{
    fn default() -> Self {
        Self { 
            state: GuiState::Setup(SetupData{
                generator_layers:vec![(3,Activation::ReLU)],
                discriminator_layers: vec![(1,Activation::ReLU)],
                loss: Loss::Squared,
                generator_inputs: 1,
                images_dir_path: String::new(),
                image_dimensions: (1,1)
            })
        }
    }
}
impl eframe::App for GenGui{
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let GuiState::Setup(setup_data) = &mut self.state{
            if let Some(running_data) = menus::setup_menu(ctx, setup_data){
                self.state = GuiState::Running(running_data);
            }
        }else if let GuiState::Running(running_data) = &mut self.state{
            if running_data.cycle_data.training{
                let mut image_data:DVector<f64>;
                match running_data.cycle_data.cycle_state{
                    CycleState::Real=>{
                        image_data = running_data.gan.generate();
                        image_data.fill(0.0);
                        image_data.iter_mut()
                            .step_by(3)
                            .for_each(|x| *x = 1.0);
                        running_data.gan.train_discriminator(image_data.clone(),running_data.discrim_pos_result.clone(), running_data.step_size);
                    },
                    CycleState::Fake=>{
                        image_data = running_data.gan.generate();
                        running_data.gan.train_discriminator(image_data.clone(),running_data.discrim_neg_result.clone(), running_data.step_size);
                    },
                    CycleState::Gen=>{
                        image_data = running_data.gan.generate();
                        running_data.gan.train_generator(running_data.discrim_pos_result.clone(), running_data.step_size);
                    }
                }
                running_data.image = Some(dvector_to_image(&running_data.image_dimensions,image_data));
                running_data.cycle_data.advance();
            }

            if let Some(setup_data) = menus::running_menu(ctx, running_data){
                self.state = GuiState::Setup(setup_data);
            }
        }
        ctx.request_repaint();
    }
}

fn csv_to_array<P: AsRef<std::path::Path>>(path:P) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    //2d vec of entries and all their data as seperate entries in vec
    let mut rdr = Reader::from_path(path)?;
    let mut entries:Vec<Vec<usize>> = Vec::new();
    for result in rdr.records() {
        let record = result?;
        entries.push(record.iter().map(|x| (*x).parse::<usize>().expect("sould contain only ints as entries")).collect());
    }
    Ok(entries)
}

//saves dvect as grayscale image
fn save_grayscale<P: AsRef<std::path::Path>>(data:DVector<f64>,path:P,width:usize,height:usize){
    let mut image = image::RgbImage::new(width as u32,height as u32);
    for (mut pixel,value) in image.pixels_mut().zip(data.iter()){
        pixel.0 = [(*value*255.0) as u8;3];
    }
    image.save(path);
}

fn array_to_csv(data:Vec<(f64,f64,f64)>) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = Writer::from_path("output.csv")?;
    for item in data{
        wtr.write_record(&[item.0.to_string(),item.1.to_string(),item.2.to_string()])?;
    }
    wtr.flush()?;
    Ok(())
}
