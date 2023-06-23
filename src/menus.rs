use eframe::egui;

use super::*;

pub fn setup_menu(ctx: &egui::Context,setup_data:&mut SetupData)->Option<RunningData>{
    let mut setup_done = false;
    egui::CentralPanel::default().show(ctx, |ui|{
        ui.heading("Set up generator and discriminator networks");
        ui.separator();
        ui.label("Network Loss");
        //display networks layers
        egui::ComboBox::from_id_source("loss combobox")
            .selected_text(format!("{:?}", setup_data.loss))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut setup_data.loss, Loss::Squared, "Squared");
                ui.selectable_value(&mut setup_data.loss, Loss::Absolute, "Absolute");
            });
        ui.label("Generator Inputs:");
        ui.add(egui::DragValue::new(&mut setup_data.generator_inputs));
        ui.separator();
        ui.horizontal(|ui|{
            ui.vertical(|ui|{
                ui.label("Generator:");
                layer_edit_menu(ui,"gen", &mut setup_data.generator_layers);
            });
            ui.separator();
            ui.vertical(|ui|{
                ui.label("Discriminator:");
                layer_edit_menu(ui,"disc",&mut setup_data.discriminator_layers);
            });
        });
        ui.separator();
        ui.label("Images Path");
        ui.small("! The genorators final layer will be changed to reflect the dimensions of these images.");
        ui.text_edit_singleline(&mut setup_data.images_dir_path);

        ui.horizontal(|ui|{
            ui.add(egui::DragValue::new(&mut setup_data.image_dimensions.0).prefix("X: "));
            ui.add(egui::DragValue::new(&mut setup_data.image_dimensions.1).prefix("Y: "));
        });
        ui.separator();
        setup_done = ui.button("Done").clicked();
    });
    if setup_done{
        //set last layer to it image dimensions
        setup_data.generator_layers.last_mut().unwrap().0 = setup_data.image_dimensions.0*setup_data.image_dimensions.1*3;
        //set last layer to two
        setup_data.discriminator_layers.last_mut().unwrap().0 = 2;
        Some(setup_data.to_running())
    }else{
        None
    }
}

pub fn running_menu(ctx: &egui::Context,running_data:&mut RunningData)->Option<SetupData>{
    let mut back_to_setup = false;
    egui::SidePanel::left("left_panel").show(ctx,|ui|{
        back_to_setup = ui.button("Back to Setup").clicked();
        ui.separator();
        ui.label("Cycles:");
        ui.add(egui::DragValue::new(&mut running_data.cycle_data.cycles));
        ui.label("Real Data per Cycle:");
        ui.add(egui::DragValue::new(&mut running_data.cycle_data.discrim_real_period));
        ui.label("False Data per Cycle:");
        ui.add(egui::DragValue::new(&mut running_data.cycle_data.discrim_fake_period));
        ui.label("Generator Training Iterations Per Cycle:");
        ui.add(egui::DragValue::new(&mut running_data.cycle_data.generator_period));
        ui.separator();
        running_data.cycle_data.training ^= ui.button("Toggle Training").clicked();
        if ui.button("Reset Training Progress").clicked(){
            running_data.cycle_data.state_progress = 1;
            running_data.cycle_data.state_progress = running_data.cycle_data.discrim_real_period;
            running_data.cycle_data.current_cycle = 1;
            running_data.cycle_data.cycle_state = CycleState::Real;
            running_data.cycle_data.training = false;
        }
        ui.separator();
        ui.add(egui::DragValue::new(&mut running_data.step_size).speed(0.00001));
        ui.separator();
        ui.label(format!("Training: \n{}",running_data.cycle_data.training));
        ui.label(format!("Current Cycle:\n{}/{}",running_data.cycle_data.current_cycle,running_data.cycle_data.cycles));
        ui.label(format!("Cycle State: \n{:?}",running_data.cycle_data.cycle_state));
        let limit = match running_data.cycle_data.cycle_state{
            CycleState::Real=>running_data.cycle_data.discrim_real_period,
            CycleState::Fake=>running_data.cycle_data.discrim_fake_period,
            CycleState::Gen=>running_data.cycle_data.generator_period
        };
        ui.label(format!("{}/{}",1+limit-running_data.cycle_data.state_progress,limit));
        ui.separator();
        if ui.button("Generate").clicked(){
            let gan_out = running_data.gan.generate();
            let out_image = dvector_to_image(&running_data.image_dimensions, gan_out);
            running_data.image = Some(out_image);
        }
    });
    egui::CentralPanel::default().show(ctx,|ui|{
        ui.label("Image:");
        if let Some(image) = &running_data.image{
            let handle = ui.ctx().load_texture("gan out", image.clone(), Default::default());
            ui.image(handle.id(), handle.size_vec2()*(ui.available_height()/handle.size()[1] as f32));
        }
    });
    if back_to_setup{
        Some(running_data.setup_data.clone())
    }else{
        None
    }
}

pub fn dvector_to_image(dimensions:&(usize,usize),data:DVector<f64>)->egui::ColorImage{
    let data_colours:Vec<egui::Color32> =  data.data.as_vec()
        .chunks(3)
        .map(|chunk| {
            let rgb:Vec<u8> = chunk.into_iter()
                .map(|x| u8::clamp((x*255.0) as u8, 0, 255))
                .collect();
            egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        })
        .collect();
    let mut out_image = egui::ColorImage::new([dimensions.0,dimensions.1], egui::Color32::RED);

    for x in 0..out_image.width(){
        for y in 0.. out_image.height(){
            out_image[(x,y)] = data_colours[(y*out_image.width()) + x];
        }
    }
    out_image
}

pub fn layer_edit_menu(ui: &mut egui::Ui,id:&str,layers: &mut Vec<(usize,Activation)>){
    let last_layer = layers.last_mut().unwrap();

    ui.horizontal(|ui|{
        
    });

    ui.horizontal(|ui|{
        ui.vertical(|ui|{
            ui.label("Last Layer Activation:");
            egui::ComboBox::from_id_source(id)
                .selected_text(format!("{:?}", last_layer.1))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut last_layer.1, Activation::Tanh, "Tanh");
                    ui.selectable_value(&mut last_layer.1, Activation::Sigmoid, "Sigmoid");
                    ui.selectable_value(&mut last_layer.1, Activation::ReLU, "ReLU");
                    ui.selectable_value(&mut last_layer.1, Activation::LeakyReLU(0.001), "LeakyReLU");
                });
        });
        ui.vertical(|ui|{
            ui.label("Last Layer Size:");
            ui.add(egui::DragValue::new(&mut last_layer.0).clamp_range(1..=usize::max_value()));
        });
    });

    ui.horizontal(|ui|{
        if ui.button("Add").clicked(){
            let last_layer = layers.last().unwrap().clone();
            layers.push(last_layer);
        }
        if ui.button("Remove").clicked() && layers.len()>1{
            layers.pop();
        }
    });
}
