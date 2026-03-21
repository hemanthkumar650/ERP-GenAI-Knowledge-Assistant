using dotnet_worker.Services;

namespace dotnet_worker.Workers;

public class PolicyUpdateWorker : BackgroundService
{
    private readonly PoliciesService _policiesService;
    private readonly EmbeddingService _embeddingService;
    private readonly IConfiguration _configuration;
    private readonly ILogger<PolicyUpdateWorker> _logger;
    private string? _lastSnapshot;

    public PolicyUpdateWorker(
        PoliciesService policiesService,
        EmbeddingService embeddingService,
        IConfiguration configuration,
        ILogger<PolicyUpdateWorker> logger)
    {
        _policiesService = policiesService;
        _embeddingService = embeddingService;
        _configuration = configuration;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var pollIntervalSeconds = _configuration.GetValue("Worker:PollIntervalSeconds", 10);
        var delay = TimeSpan.FromSeconds(Math.Max(3, pollIntervalSeconds));

        _logger.LogInformation("PolicyUpdateWorker started. Watching {Path}", _policiesService.GetPoliciesPath());

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var currentSnapshot = _policiesService.BuildSnapshot();

                if (_lastSnapshot is null)
                {
                    _lastSnapshot = currentSnapshot;
                    _logger.LogInformation("Initial policy snapshot captured.");
                }
                else if (!string.Equals(_lastSnapshot, currentSnapshot, StringComparison.Ordinal))
                {
                    _logger.LogInformation("Policy change detected. Triggering reindex.");
                    var refreshed = await _embeddingService.RefreshEmbeddingsAsync(stoppingToken);
                    if (refreshed)
                    {
                        _lastSnapshot = currentSnapshot;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Policy update loop failed.");
            }

            await Task.Delay(delay, stoppingToken);
        }
    }
}
