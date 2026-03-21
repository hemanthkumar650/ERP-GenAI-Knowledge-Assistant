using dotnet_worker.Services;
using dotnet_worker.Workers;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddHttpClient<ChromaService>();
builder.Services.AddSingleton<PoliciesService>();
builder.Services.AddSingleton<EmbeddingService>();
builder.Services.AddHostedService<PolicyUpdateWorker>();

var host = builder.Build();
host.Run();
