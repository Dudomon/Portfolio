defmodule Marketplace.MixProject do
  use Mix.Project

  def project do
    [
      app: :marketplace,
      version: "0.1.0",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.html": :test
      ]
    ]
  end

  def application do
    [
      mod: {Marketplace.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  # Compile paths vary by environment. Test environment includes test/support
  # for factory modules and test helpers without polluting production builds.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Phoenix core. Version 1.7 introduces verified routes eliminating
      # runtime route helper errors.
      {:phoenix, "~> 1.7.10"},
      {:phoenix_ecto, "~> 4.4"},
      {:phoenix_html, "~> 3.3"},
      {:phoenix_live_reload, "~> 1.4", only: :dev},
      {:phoenix_live_view, "~> 0.20"},
      {:phoenix_live_dashboard, "~> 0.8"},

      # Ecto with PostgreSQL. Postgres chosen over MySQL for JSONB support
      # (product attributes vary by category) and better concurrent index builds.
      {:ecto_sql, "~> 3.11"},
      {:postgrex, "~> 0.17"},

      # Absinthe GraphQL. Relay conventions disabled; they add complexity
      # without benefit for this use case.
      {:absinthe, "~> 1.7"},
      {:absinthe_plug, "~> 1.5"},
      {:absinthe_phoenix, "~> 2.0"},
      {:dataloader, "~> 2.0"},

      # Authentication. Bcrypt over Argon2 for broader deployment compatibility.
      # Security difference is negligible for this threat model.
      {:bcrypt_elixir, "~> 3.1"},
      {:guardian, "~> 2.3"},

      # Decimal arithmetic for money. Never use floats for currency.
      {:decimal, "~> 2.1"},

      # JSON encoding. Jason outperforms Poison by 2x on encoding.
      {:jason, "~> 1.4"},

      # HTTP client for external payment gateway integration.
      {:finch, "~> 0.17"},

      # Telemetry for metrics. Phoenix and Ecto emit telemetry events
      # by default; this captures them.
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},

      # Development and test dependencies
      {:floki, "~> 0.35", only: :test},
      {:ex_machina, "~> 2.7", only: :test},
      {:excoveralls, "~> 0.18", only: :test},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "ecto.setup"],
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"]
    ]
  end
end
