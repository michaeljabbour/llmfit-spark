mod display;
mod theme;
mod tui_app;
mod tui_events;
mod tui_ui;

use clap::{Parser, Subcommand};
use llmfit_core::cluster::ClusterSpecs;
use llmfit_core::fit::ModelFit;
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;

#[derive(Parser)]
#[command(name = "llmfit")]
#[command(about = "Right-size LLM models to your system's hardware", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Show only models that perfectly match recommended specs
    #[arg(short, long)]
    perfect: bool,

    /// Limit number of results
    #[arg(short = 'n', long)]
    limit: Option<usize>,

    /// Use classic CLI table output instead of TUI
    #[arg(long)]
    cli: bool,

    /// Output results as JSON (for tool integration)
    #[arg(long)]
    json: bool,

    /// Override GPU VRAM size (e.g. "32G", "32000M", "1.5T").
    /// Useful when GPU memory autodetection fails.
    #[arg(long, value_name = "SIZE")]
    memory: Option<String>,

    /// Use cluster mode (auto-loads saved cluster config).
    /// Overrides local hardware detection with cluster resources.
    #[arg(long)]
    cluster: bool,

    /// Disable cluster mode even if a cluster config exists.
    #[arg(long)]
    no_cluster: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Show system hardware specifications
    System,

    /// List all available LLM models
    List,

    /// Find models that fit your system (classic table output)
    Fit {
        /// Show only models that perfectly match recommended specs
        #[arg(short, long)]
        perfect: bool,

        /// Limit number of results
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Search for specific models
    Search {
        /// Search query (model name, provider, or size)
        query: String,
    },

    /// Show detailed information about a specific model
    Info {
        /// Model name or partial name to look up
        model: String,
    },

    /// Recommend top models for your hardware (JSON-friendly)
    Recommend {
        /// Limit number of recommendations
        #[arg(short = 'n', long, default_value = "5")]
        limit: usize,

        /// Filter by use case: general, coding, reasoning, chat, multimodal, embedding
        #[arg(long, value_name = "CATEGORY")]
        use_case: Option<String>,

        /// Filter by minimum fit level: perfect, good, marginal
        #[arg(long, default_value = "marginal")]
        min_fit: String,

        /// Filter by inference runtime: mlx, llamacpp, vllm, any
        #[arg(long, default_value = "any")]
        runtime: String,

        /// Output as JSON (default for recommend)
        #[arg(long, default_value = "true")]
        json: bool,
    },

    /// Benchmark inference speed (TPS, TTFT, latency)
    Bench {
        /// Model name or partial match (auto-detects if omitted)
        #[arg(long)]
        model: Option<String>,

        /// Provider: ollama, vllm, mlx, or auto
        #[arg(long, default_value = "auto")]
        provider: String,

        /// Base URL override (e.g. http://203.0.113.10:8000); falls back to $LLMFIT_VLLM_URL
        #[arg(long)]
        url: Option<String>,

        /// Number of benchmark runs per model
        #[arg(long, default_value = "3")]
        runs: usize,

        /// Benchmark all available models across all providers
        #[arg(long)]
        all: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Manage DGX Spark cluster configuration
    Cluster {
        #[command(subcommand)]
        action: ClusterAction,
    },
}

#[derive(Subcommand)]
enum ClusterAction {
    /// Initialize cluster config (interactive setup)
    Init,
    /// Show current cluster configuration and status
    Status,
    /// Remove saved cluster configuration
    Remove,
}

/// Detect system specs, with optional cluster mode and memory override.
fn detect_specs(
    memory_override: &Option<String>,
    use_cluster: bool,
    no_cluster: bool,
) -> SystemSpecs {
    // Cluster mode: use saved cluster config if available
    if !no_cluster && (use_cluster || ClusterSpecs::load().is_some()) {
        if let Some(cluster) = ClusterSpecs::load() {
            if use_cluster || !no_cluster {
                return cluster.to_system_specs();
            }
        }
    }

    // Local detection
    let specs = SystemSpecs::detect();
    if let Some(mem_str) = memory_override {
        match llmfit_core::hardware::parse_memory_size(mem_str) {
            Some(gb) => specs.with_gpu_memory_override(gb),
            None => {
                eprintln!(
                    "Warning: could not parse --memory value '{}'. Expected format: 32G, 32000M, 1.5T",
                    mem_str
                );
                specs
            }
        }
    } else {
        specs
    }
}

fn run_fit(
    perfect: bool,
    limit: Option<usize>,
    json: bool,
    memory_override: &Option<String>,
    use_cluster: bool,
    no_cluster: bool,
) {
    let specs = detect_specs(memory_override, use_cluster, no_cluster);
    let db = ModelDatabase::new();

    if !json {
        if specs.cluster_mode {
            if let Some(cluster) = ClusterSpecs::load() {
                cluster.display();
            }
        } else {
            specs.display();
        }
    }

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .map(|m| ModelFit::analyze(m, &specs))
        .collect();

    if perfect {
        fits.retain(|f| f.fit_level == llmfit_core::fit::FitLevel::Perfect);
    }

    fits = llmfit_core::fit::rank_models_by_fit(fits);

    if let Some(n) = limit {
        fits.truncate(n);
    }

    if json {
        display::display_json_fits(&specs, &fits);
    } else {
        display::display_model_fits(&fits);
    }
}

fn run_tui(
    memory_override: &Option<String>,
    use_cluster: bool,
    no_cluster: bool,
) -> std::io::Result<()> {
    // Setup terminal
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend)?;

    // Create app state
    let specs = detect_specs(memory_override, use_cluster, no_cluster);
    let mut app = tui_app::App::with_specs(specs);

    // Main loop
    loop {
        terminal.draw(|frame| {
            tui_ui::draw(frame, &mut app);
        })?;

        tui_events::handle_events(&mut app)?;

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn run_recommend(
    limit: usize,
    use_case: Option<String>,
    min_fit: String,
    runtime_filter: String,
    json: bool,
    memory_override: &Option<String>,
    use_cluster: bool,
    no_cluster: bool,
) {
    let specs = detect_specs(memory_override, use_cluster, no_cluster);
    let db = ModelDatabase::new();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .map(|m| ModelFit::analyze(m, &specs))
        .collect();

    // Filter by minimum fit level
    let min_level = match min_fit.to_lowercase().as_str() {
        "perfect" => llmfit_core::fit::FitLevel::Perfect,
        "good" => llmfit_core::fit::FitLevel::Good,
        "marginal" => llmfit_core::fit::FitLevel::Marginal,
        _ => llmfit_core::fit::FitLevel::Marginal,
    };
    fits.retain(|f| match (min_level, f.fit_level) {
        (llmfit_core::fit::FitLevel::Marginal, llmfit_core::fit::FitLevel::TooTight) => false,
        (
            llmfit_core::fit::FitLevel::Good,
            llmfit_core::fit::FitLevel::TooTight | llmfit_core::fit::FitLevel::Marginal,
        ) => false,
        (llmfit_core::fit::FitLevel::Perfect, llmfit_core::fit::FitLevel::Perfect) => true,
        (llmfit_core::fit::FitLevel::Perfect, _) => false,
        _ => true,
    });

    // Filter by runtime
    match runtime_filter.to_lowercase().as_str() {
        "mlx" => fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::Mlx),
        "llamacpp" | "llama.cpp" | "llama_cpp" => {
            fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::LlamaCpp)
        }
        "vllm" => fits.retain(|f| f.runtime == llmfit_core::fit::InferenceRuntime::VLlm),
        _ => {} // "any" or unrecognized — keep all
    }

    // Filter by use case if specified
    if let Some(ref uc) = use_case {
        let target = match uc.to_lowercase().as_str() {
            "coding" | "code" => Some(llmfit_core::models::UseCase::Coding),
            "reasoning" | "reason" => Some(llmfit_core::models::UseCase::Reasoning),
            "chat" => Some(llmfit_core::models::UseCase::Chat),
            "multimodal" | "vision" => Some(llmfit_core::models::UseCase::Multimodal),
            "embedding" | "embed" => Some(llmfit_core::models::UseCase::Embedding),
            "general" => Some(llmfit_core::models::UseCase::General),
            _ => None,
        };
        if let Some(target_uc) = target {
            fits.retain(|f| f.use_case == target_uc);
        }
    }

    fits = llmfit_core::fit::rank_models_by_fit(fits);
    fits.truncate(limit);

    if json {
        display::display_json_fits(&specs, &fits);
    } else {
        if !fits.is_empty() {
            if specs.cluster_mode {
                if let Some(cluster) = ClusterSpecs::load() {
                    cluster.display();
                }
            } else {
                specs.display();
            }
        }
        display::display_model_fits(&fits);
    }
}

fn target_info(target: &llmfit_core::bench::BenchTarget) -> (&str, &str, &str) {
    use llmfit_core::bench::BenchTarget;
    match target {
        BenchTarget::Ollama { url, model } => ("Ollama", url.as_str(), model.as_str()),
        BenchTarget::VLlm { url, model } => ("vLLM", url.as_str(), model.as_str()),
        BenchTarget::Mlx { url, model } => ("MLX", url.as_str(), model.as_str()),
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

fn run_bench(model: Option<String>, provider: String, url_override: Option<String>, runs: usize, all: bool, json: bool) {
    use llmfit_core::bench;

    // --all mode: discover and bench every available model
    if all {
        let targets = bench::discover_all_targets();
        if targets.is_empty() {
            eprintln!("No providers or models found. Start Ollama, vLLM, or MLX first.");
            std::process::exit(1);
        }

        if !json {
            println!();
            println!("  Found {} model(s) across all providers:", targets.len());
            for t in &targets {
                let (prov, _, mdl) = target_info(t);
                println!("    - {} ({})", mdl, prov);
            }
            println!();
        }

        let mut results = Vec::new();
        for target in &targets {
            let (provider_name, url, model_name) = target_info(target);
            if !json {
                println!("  ─── {} via {} ───", model_name, provider_name);
            }
            let progress = |i: usize, total: usize| {
                if !json {
                    if i == 0 {
                        eprint!("  Warming up...");
                    } else {
                        eprint!("\r  Run {}/{}...", i, total);
                    }
                }
            };

            let result = match target {
                bench::BenchTarget::Ollama { url, model } => {
                    bench::bench_ollama(url, model, runs, &progress)
                }
                bench::BenchTarget::VLlm { url, model } => {
                    bench::bench_openai_compat(url, model, "vllm", runs, &progress)
                }
                bench::BenchTarget::Mlx { url, model } => {
                    bench::bench_openai_compat(url, model, "mlx", runs, &progress)
                }
            };

            if !json {
                eprintln!();
            }

            match result {
                Ok(r) => {
                    if !json {
                        r.display();
                    }
                    results.push(r);
                }
                Err(e) => {
                    if !json {
                        eprintln!("  Error: {}\n", e);
                    }
                }
            }
        }

        if json {
            let json_out = serde_json::json!({
                "benchmarks": results.iter().map(|r| serde_json::json!({
                    "model": r.model,
                    "provider": r.provider,
                    "summary": r.summary,
                    "runs": r.runs,
                })).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&json_out).unwrap());
        } else if results.len() > 1 {
            // Print comparison table
            println!("  ═══ Comparison ═══");
            println!();
            println!("  {:30} {:>8} {:>10} {:>10} {:>8}", "Model", "Provider", "TPS avg", "TTFT avg", "Latency");
            println!("  {:30} {:>8} {:>10} {:>10} {:>8}", "─".repeat(30), "────────", "──────────", "──────────", "────────");
            for r in &results {
                println!(
                    "  {:30} {:>8} {:>9.1}  {:>8.0}ms {:>6.0}ms",
                    truncate_str(&r.model, 30),
                    r.provider,
                    r.summary.avg_tps,
                    r.summary.avg_ttft_ms,
                    r.summary.avg_total_ms,
                );
            }
            println!();
        }

        return;
    }

    let target = match provider.to_lowercase().as_str() {
        "ollama" => {
            let url = url_override.clone().unwrap_or_else(||
                std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string()));
            let model_name = model.unwrap_or_else(|| {
                eprintln!("Error: --model required for ollama provider");
                std::process::exit(1);
            });
            bench::BenchTarget::Ollama { url, model: model_name }
        }
        "vllm" => {
            let url = url_override.clone().unwrap_or_else(||
                std::env::var("LLMFIT_VLLM_URL").unwrap_or_else(|_|
                    if let Some(cluster) = ClusterSpecs::load() {
                        format!("http://{}:8000", cluster.head_ip)
                    } else {
                        "http://localhost:8000".to_string()
                    }));
            // Try to auto-detect model from the vLLM endpoint
            match bench::detect_model_from_url(&url, model.as_deref()) {
                Ok(model_name) => bench::BenchTarget::VLlm { url, model: model_name },
                Err(_) => {
                    let model_name = model.unwrap_or_else(|| {
                        eprintln!("Error: could not detect model from vLLM at {}. Use --model", url);
                        std::process::exit(1);
                    });
                    bench::BenchTarget::VLlm { url, model: model_name }
                }
            }
        }
        "mlx" => {
            let url = url_override.clone().unwrap_or_else(||
                std::env::var("MLX_LM_HOST")
                    .unwrap_or_else(|_| "http://localhost:8080".to_string()));
            let model_name = model.unwrap_or_else(|| {
                eprintln!("Error: --model required for mlx provider");
                std::process::exit(1);
            });
            bench::BenchTarget::Mlx { url, model: model_name }
        }
        "auto" | _ => {
            match bench::auto_detect_target(model.as_deref()) {
                Ok(target) => target,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    };

    // Show what we're benchmarking
    let (provider_name, url, model_name) = match &target {
        bench::BenchTarget::Ollama { url, model } => ("Ollama", url.as_str(), model.as_str()),
        bench::BenchTarget::VLlm { url, model } => ("vLLM", url.as_str(), model.as_str()),
        bench::BenchTarget::Mlx { url, model } => ("MLX", url.as_str(), model.as_str()),
    };

    if !json {
        println!();
        println!("  Benchmarking {} via {} ({})", model_name, provider_name, url);
        println!("  {} run(s) with warmup...", runs);
        println!();
    }

    let progress = |i: usize, total: usize| {
        if !json {
            if i == 0 {
                eprint!("  Warming up...");
            } else {
                eprint!("\r  Run {}/{}...", i, total);
            }
        }
    };

    let result = match target {
        bench::BenchTarget::Ollama { ref url, ref model } => {
            bench::bench_ollama(url, model, runs, &progress)
        }
        bench::BenchTarget::VLlm { ref url, ref model } => {
            bench::bench_openai_compat(url, model, "vllm", runs, &progress)
        }
        bench::BenchTarget::Mlx { ref url, ref model } => {
            bench::bench_openai_compat(url, model, "mlx", runs, &progress)
        }
    };

    if !json {
        eprintln!(); // clear progress line
    }

    match result {
        Ok(result) => {
            if json {
                result.display_json();
            } else {
                result.display();
            }
        }
        Err(e) => {
            eprintln!("Benchmark failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    // If a subcommand is given, use classic CLI mode
    if let Some(command) = cli.command {
        match command {
            Commands::System => {
                let specs = detect_specs(&cli.memory, cli.cluster, cli.no_cluster);
                if specs.cluster_mode {
                    if let Some(cluster) = ClusterSpecs::load() {
                        if cli.json {
                            cluster.display_json();
                        } else {
                            cluster.display();
                        }
                    }
                } else if cli.json {
                    display::display_json_system(&specs);
                } else {
                    specs.display();
                }
            }

            Commands::List => {
                let db = ModelDatabase::new();
                display::display_all_models(db.get_all_models());
            }

            Commands::Fit { perfect, limit } => {
                run_fit(perfect, limit, cli.json, &cli.memory, cli.cluster, cli.no_cluster);
            }

            Commands::Search { query } => {
                let db = ModelDatabase::new();
                let results = db.find_model(&query);
                display::display_search_results(&results, &query);
            }

            Commands::Info { model } => {
                let db = ModelDatabase::new();
                let specs = detect_specs(&cli.memory, cli.cluster, cli.no_cluster);
                let results = db.find_model(&model);

                if results.is_empty() {
                    println!("\nNo model found matching '{}'", model);
                    return;
                }

                if results.len() > 1 {
                    println!("\nMultiple models found. Please be more specific:");
                    for m in results {
                        println!("  - {}", m.name);
                    }
                    return;
                }

                let fit = ModelFit::analyze(results[0], &specs);
                if cli.json {
                    display::display_json_fits(&specs, &[fit]);
                } else {
                    display::display_model_detail(&fit);
                }
            }

            Commands::Recommend {
                limit,
                use_case,
                min_fit,
                runtime,
                json,
            } => {
                run_recommend(
                    limit,
                    use_case,
                    min_fit,
                    runtime,
                    json,
                    &cli.memory,
                    cli.cluster,
                    cli.no_cluster,
                );
            }

            Commands::Bench {
                model,
                provider,
                url,
                runs,
                all,
                json,
            } => {
                run_bench(model, provider, url, runs, all, json || cli.json);
            }

            Commands::Cluster { action } => {
                match action {
                    ClusterAction::Init => {
                        match llmfit_core::cluster::interactive_init() {
                            Ok(cluster) => {
                                println!(
                                    "Cluster configured: {} nodes, {:.0} GB total VRAM",
                                    cluster.node_count(),
                                    cluster.total_vram_gb()
                                );
                                println!("Run `llmfit` to see models sized for your cluster.");
                            }
                            Err(e) => {
                                eprintln!("Cluster init failed: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    ClusterAction::Status => {
                        match ClusterSpecs::load() {
                            Some(cluster) => {
                                if cli.json {
                                    cluster.display_json();
                                } else {
                                    cluster.display();
                                    // Check if Ray is reachable
                                    if cluster.is_ray_reachable() {
                                        println!("  Ray Dashboard: ONLINE ({}:{})", cluster.head_ip, cluster.ray_port);
                                    } else {
                                        println!("  Ray Dashboard: OFFLINE ({}:{})", cluster.head_ip, cluster.ray_port);
                                    }
                                    println!();
                                }
                            }
                            None => {
                                println!("No cluster configured. Run `llmfit cluster init` to set up.");
                            }
                        }
                    }
                    ClusterAction::Remove => {
                        match ClusterSpecs::remove_config() {
                            Ok(()) => println!("Cluster configuration removed."),
                            Err(e) => {
                                eprintln!("Failed to remove cluster config: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // If --cli flag, use classic fit output
    if cli.cli {
        run_fit(cli.perfect, cli.limit, cli.json, &cli.memory, cli.cluster, cli.no_cluster);
        return;
    }

    // Default: launch TUI
    if let Err(e) = run_tui(&cli.memory, cli.cluster, cli.no_cluster) {
        eprintln!("Error running TUI: {}", e);
        std::process::exit(1);
    }
}
