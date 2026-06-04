use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    MacInt8,
    CudaFp8,
    CudaNf4,
    RunaiH200Fp8,
}

impl Backend {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "mac-int8" | "mac" | "mps" => Some(Self::MacInt8),
            "cuda-fp8" | "fp8" => Some(Self::CudaFp8),
            "cuda-nf4" | "nf4" => Some(Self::CudaNf4),
            "runai-h200-fp8" | "h200" | "runai" => Some(Self::RunaiH200Fp8),
            _ => None,
        }
    }

    fn script(self) -> &'static str {
        match self {
            Self::MacInt8 => "scripts/smoke_mac_mps_int8.sh",
            Self::CudaFp8 => "scripts/smoke_cuda_fp8.sh",
            Self::CudaNf4 => "scripts/smoke_cuda_nf4.sh",
            Self::RunaiH200Fp8 => "scripts/smoke_runai_h200_fp8.sh",
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::MacInt8 => "mac-int8",
            Self::CudaFp8 => "cuda-fp8",
            Self::CudaNf4 => "cuda-nf4",
            Self::RunaiH200Fp8 => "runai-h200-fp8",
        }
    }
}

#[derive(Clone, Debug)]
struct Settings {
    backend: Backend,
    width: u32,
    height: u32,
    preset: String,
    seed: u64,
    out_dir: PathBuf,
    repo: PathBuf,
}

impl Settings {
    fn default(repo: PathBuf) -> Self {
        Self {
            backend: Backend::MacInt8,
            width: 512,
            height: 512,
            preset: "V4_TURBO_12".to_string(),
            seed: 0,
            out_dir: PathBuf::from("outputs/rust-chat"),
            repo,
        }
    }
}

enum PromptSource {
    Text(String),
    File(PathBuf),
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo = find_repo_root()?;
    let mut settings = Settings::default(repo);
    let mut args = env::args().skip(1).collect::<Vec<_>>();

    if args.is_empty() {
        print_help();
        return Ok(());
    }

    let command = args.remove(0);
    match command.as_str() {
        "chat" => parse_global_flags(&mut settings, &args).and_then(|_| chat(settings)),
        "generate" | "gen" => generate_command(settings, &args),
        "check" => {
            println!("repo: {}", settings.repo.display());
            run_check_env(&settings)
        }
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        other => Err(format!(
            "unknown command `{other}`; run `ideogram4-chat help`"
        )),
    }
}

fn generate_command(mut settings: Settings, args: &[String]) -> Result<(), String> {
    let mut prompt: Option<PromptSource> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => {
                i += 1;
                settings.backend = parse_backend_arg(args.get(i))?;
            }
            "--width" => {
                i += 1;
                settings.width = parse_u32(args.get(i), "--width")?;
            }
            "--height" => {
                i += 1;
                settings.height = parse_u32(args.get(i), "--height")?;
            }
            "--size" => {
                i += 1;
                settings.width = parse_u32(args.get(i), "--size width")?;
                i += 1;
                settings.height = parse_u32(args.get(i), "--size height")?;
            }
            "--preset" => {
                i += 1;
                settings.preset = require_value(args.get(i), "--preset")?.to_string();
            }
            "--seed" => {
                i += 1;
                settings.seed = parse_u64(args.get(i), "--seed")?;
            }
            "--out-dir" => {
                i += 1;
                settings.out_dir = PathBuf::from(require_value(args.get(i), "--out-dir")?);
            }
            "--output" | "-o" => {
                i += 1;
                output = Some(PathBuf::from(require_value(args.get(i), "--output")?));
            }
            "--prompt" | "-p" => {
                i += 1;
                prompt = Some(PromptSource::Text(
                    require_value(args.get(i), "--prompt")?.to_string(),
                ));
            }
            "--prompt-file" | "-f" => {
                i += 1;
                prompt = Some(PromptSource::File(PathBuf::from(require_value(
                    args.get(i),
                    "--prompt-file",
                )?)));
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other if !other.starts_with('-') && prompt.is_none() => {
                prompt = Some(PromptSource::Text(other.to_string()));
            }
            other => return Err(format!("unknown option `{other}`")),
        }
        i += 1;
    }

    let prompt = prompt.ok_or("missing --prompt or --prompt-file")?;
    let output = output.unwrap_or_else(|| default_output_path(&settings, &prompt));
    run_generation(&settings, prompt, output)
}

fn chat(mut settings: Settings) -> Result<(), String> {
    println!("ideogram4-chat");
    println!(
        "backend={} size={}x{} preset={} seed={}",
        settings.backend.as_str(),
        settings.width,
        settings.height,
        settings.preset,
        settings.seed
    );
    println!(
        "Type a prompt to generate. Commands: /backend, /size, /preset, /seed, /out, /file, /check, /quit"
    );

    let stdin = io::stdin();
    loop {
        print!("ideogram4> ");
        io::stdout().flush().map_err(|e| e.to_string())?;

        let mut line = String::new();
        let n = stdin.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            println!();
            return Ok(());
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('/') {
            if handle_chat_command(rest, &mut settings)? {
                return Ok(());
            }
            continue;
        }

        let output = default_output_path(&settings, &PromptSource::Text(line.to_string()));
        run_generation(&settings, PromptSource::Text(line.to_string()), output)?;
    }
}

fn handle_chat_command(command: &str, settings: &mut Settings) -> Result<bool, String> {
    let mut parts = command.split_whitespace().collect::<Vec<_>>();
    if parts.is_empty() {
        return Ok(false);
    }

    match parts.remove(0) {
        "q" | "quit" | "exit" => Ok(true),
        "help" => {
            println!("/backend mac-int8|cuda-fp8|cuda-nf4|runai-h200-fp8");
            println!("/size WIDTH HEIGHT");
            println!("/preset V4_TURBO_12|V4_DEFAULT_20|V4_QUALITY_48");
            println!("/seed N");
            println!("/out DIR");
            println!("/file PATH");
            println!("/check");
            println!("/quit");
            Ok(false)
        }
        "backend" => {
            let value = parts.first().ok_or("usage: /backend <backend>")?;
            settings.backend =
                Backend::parse(value).ok_or_else(|| format!("unknown backend `{value}`"))?;
            println!("backend={}", settings.backend.as_str());
            Ok(false)
        }
        "size" => {
            if parts.len() != 2 {
                return Err("usage: /size <width> <height>".to_string());
            }
            settings.width = parts[0].parse().map_err(|_| "invalid width".to_string())?;
            settings.height = parts[1].parse().map_err(|_| "invalid height".to_string())?;
            println!("size={}x{}", settings.width, settings.height);
            Ok(false)
        }
        "preset" => {
            let value = parts.first().ok_or("usage: /preset <preset>")?;
            settings.preset = (*value).to_string();
            println!("preset={}", settings.preset);
            Ok(false)
        }
        "seed" => {
            let value = parts.first().ok_or("usage: /seed <n>")?;
            settings.seed = value.parse().map_err(|_| "invalid seed".to_string())?;
            println!("seed={}", settings.seed);
            Ok(false)
        }
        "out" => {
            let value = parts.first().ok_or("usage: /out <dir>")?;
            settings.out_dir = PathBuf::from(value);
            println!("out_dir={}", settings.out_dir.display());
            Ok(false)
        }
        "file" => {
            let value = parts.first().ok_or("usage: /file <caption.json>")?;
            let source = PromptSource::File(PathBuf::from(value));
            let output = default_output_path(settings, &source);
            run_generation(settings, source, output)?;
            Ok(false)
        }
        "check" => {
            run_check_env(settings)?;
            Ok(false)
        }
        other => Err(format!("unknown command `/{other}`")),
    }
}

fn run_generation(
    settings: &Settings,
    prompt: PromptSource,
    output: PathBuf,
) -> Result<(), String> {
    let caption = match &prompt {
        PromptSource::Text(text) => structured_caption(text),
        PromptSource::File(path) => fs::read_to_string(resolve_path(&settings.repo, path))
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?,
    };

    let script = settings.repo.join(settings.backend.script());
    if !script.exists() {
        return Err(format!("runner script not found: {}", script.display()));
    }

    let output_abs = resolve_path(&settings.repo, &output);
    if let Some(parent) = output_abs.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }

    println!(
        "[job] backend={} size={}x{} preset={} seed={} output={}",
        settings.backend.as_str(),
        settings.width,
        settings.height,
        settings.preset,
        settings.seed,
        output_abs.display()
    );

    let status = Command::new(&script)
        .current_dir(&settings.repo)
        .env("IDEOGRAM_CAPTION", caption)
        .env("IDEOGRAM_HEIGHT", settings.height.to_string())
        .env("IDEOGRAM_WIDTH", settings.width.to_string())
        .env("IDEOGRAM_PRESET", &settings.preset)
        .env("IDEOGRAM_SEED", settings.seed.to_string())
        .env("IDEOGRAM_OUTPUT", &output_abs)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run {}: {e}", script.display()))?;

    if status.success() {
        println!("[done] {}", output_abs.display());
        Ok(())
    } else {
        Err(format!("runner exited with status {status}"))
    }
}

fn run_check_env(settings: &Settings) -> Result<(), String> {
    let script = settings.repo.join("scripts/check_env.sh");
    let status = Command::new(&script)
        .current_dir(&settings.repo)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run {}: {e}", script.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("check_env exited with status {status}"))
    }
}

fn structured_caption(prompt: &str) -> String {
    let escaped = json_escape(prompt);
    format!(
        "{{\"high_level_description\":\"{escaped}\",\"compositional_deconstruction\":{{\"background\":\"scene described by the Vietnamese prompt\",\"elements\":[{{\"type\":\"obj\",\"desc\":\"{escaped}\"}}]}}}}"
    )
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn default_output_path(settings: &Settings, prompt: &PromptSource) -> PathBuf {
    let name = match prompt {
        PromptSource::File(path) => path
            .file_stem()
            .and_then(OsStr::to_str)
            .map(slugify)
            .unwrap_or_else(|| "prompt".to_string()),
        PromptSource::Text(text) => slugify(text),
    };
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    settings.out_dir.join(format!("{stamp}-{name}.png"))
}

fn slugify(value: &str) -> String {
    let mut out = String::new();
    for c in value.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if (c.is_whitespace() || c == '-' || c == '_') && !out.ends_with('-') {
            out.push('-');
        }
        if out.len() >= 48 {
            break;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "prompt".to_string()
    } else {
        trimmed
    }
}

fn find_repo_root() -> Result<PathBuf, String> {
    if let Ok(path) = env::var("IDEOGRAM_REPO") {
        return Ok(PathBuf::from(path));
    }

    let mut dir = env::current_dir().map_err(|e| e.to_string())?;
    loop {
        if dir.join("scripts/smoke_mac_mps_int8.sh").exists() && dir.join("pyproject.toml").exists()
        {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    Err("could not find repo root; run from ideogram4-test or set IDEOGRAM_REPO".to_string())
}

fn resolve_path(repo: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo.join(path)
    }
}

fn parse_global_flags(settings: &mut Settings, args: &[String]) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => {
                i += 1;
                settings.backend = parse_backend_arg(args.get(i))?;
            }
            "--width" => {
                i += 1;
                settings.width = parse_u32(args.get(i), "--width")?;
            }
            "--height" => {
                i += 1;
                settings.height = parse_u32(args.get(i), "--height")?;
            }
            "--size" => {
                i += 1;
                settings.width = parse_u32(args.get(i), "--size width")?;
                i += 1;
                settings.height = parse_u32(args.get(i), "--size height")?;
            }
            "--preset" => {
                i += 1;
                settings.preset = require_value(args.get(i), "--preset")?.to_string();
            }
            "--seed" => {
                i += 1;
                settings.seed = parse_u64(args.get(i), "--seed")?;
            }
            "--out-dir" => {
                i += 1;
                settings.out_dir = PathBuf::from(require_value(args.get(i), "--out-dir")?);
            }
            other => return Err(format!("unknown option `{other}`")),
        }
        i += 1;
    }
    Ok(())
}

fn parse_backend_arg(value: Option<&String>) -> Result<Backend, String> {
    let value = require_value(value, "--backend")?;
    Backend::parse(value).ok_or_else(|| format!("unknown backend `{value}`"))
}

fn parse_u32(value: Option<&String>, name: &str) -> Result<u32, String> {
    require_value(value, name)?
        .parse()
        .map_err(|_| format!("invalid value for {name}"))
}

fn parse_u64(value: Option<&String>, name: &str) -> Result<u64, String> {
    require_value(value, name)?
        .parse()
        .map_err(|_| format!("invalid value for {name}"))
}

fn require_value<'a>(value: Option<&'a String>, name: &str) -> Result<&'a str, String> {
    value
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {name}"))
}

fn print_help() {
    println!(
        "ideogram4-chat

Usage:
  ideogram4-chat check
  ideogram4-chat chat [--backend mac-int8] [--size 512 512]
  ideogram4-chat generate --prompt \"poster Tết chữ TẾT AN LÀNH\" [options]
  ideogram4-chat generate --prompt-file configs/prompts/vi/poster-tet-an-lanh.json [options]

Options:
  --backend mac-int8|cuda-fp8|cuda-nf4|runai-h200-fp8
  --width N
  --height N
  --size WIDTH HEIGHT
  --preset V4_TURBO_12|V4_DEFAULT_20|V4_QUALITY_48
  --seed N
  --out-dir DIR
  --output FILE

Notes:
  Rust handles terminal/API orchestration only. Python/PyTorch still runs the model.
"
    );
}
