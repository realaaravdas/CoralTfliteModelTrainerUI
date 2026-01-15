use tauri::Emitter;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use std::path::PathBuf;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[derive(serde::Deserialize)]
struct TrainingArgs {
    project_name: String,
    epochs: u32,
    batch_size: u32,
    data_path: String,
    output_dir: String,
}

#[tauri::command]
async fn start_training(app: tauri::AppHandle, args: TrainingArgs) -> Result<(), String> {
    // For this sandbox environment, we use 'python' command directly pointing to the script
    // In production with a sidecar, you would use:
    // let sidecar_command = app.shell().sidecar("train_engine").map_err(|e| e.to_string())?;

    // Construct the path to the python script relative to the current working directory or bundle
    // Assuming the python script is in "../python/train_engine.py" relative to the Tauri execution context during dev
    // But since we are running from the root in this sandbox, we might need absolute paths or careful relative paths.
    // Ideally, we pass the script path.

    // For robustness in this sandbox, let's assume we are running 'npm run tauri dev' from project root
    // But actually, the sidecar logic is cleaner.
    // Let's implement the logic to choose between sidecar and python command based on a check.

    // We will use the python command for this demo to ensure it works in the sandbox without binary packaging
    let script_path = PathBuf::from("python/train_engine.py");
    // We need absolute path because the CWD might vary
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let script_absolute_path = cwd.join(script_path);

    println!("Starting training with script: {:?}", script_absolute_path);

    let output_dir_path = PathBuf::from(&args.output_dir);
    if !output_dir_path.exists() {
        std::fs::create_dir_all(&output_dir_path).map_err(|e| e.to_string())?;
    }

    let command = app.shell().command("python3")
        .args([
            script_absolute_path.to_str().unwrap(),
            "--project-name", &args.project_name,
            "--epochs", &args.epochs.to_string(),
            "--batch-size", &args.batch_size.to_string(),
            "--data-path", &args.data_path,
            "--output-dir", &args.output_dir,
            // "--mock" // Uncomment to force mock if needed, but the script auto-detects
        ]);

    let (mut rx, child) = command.spawn().map_err(|e| e.to_string())?;

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let line_str = String::from_utf8_lossy(&line);
                    // println!("Python stdout: {}", line_str);
                    let _ = app.emit("training-log", line_str.to_string());
                }
                CommandEvent::Stderr(line) => {
                    let line_str = String::from_utf8_lossy(&line);
                    eprintln!("Python stderr: {}", line_str);
                     // Send stderr as error logs or just logs
                    let _ = app.emit("training-error", line_str.to_string());
                }
                CommandEvent::Terminated(payload) => {
                     println!("Python process terminated with code {:?}", payload.code);
                     let _ = app.emit("training-finished", payload.code);
                }
                _ => {}
            }
        }
    });

    Ok(())
}

#[tauri::command]
async fn open_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![greet, start_training, open_folder])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
