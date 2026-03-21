namespace dotnet_worker.Services;

public class ChromaService
{
    private readonly HttpClient _httpClient;
    private readonly IConfiguration _configuration;
    private readonly ILogger<ChromaService> _logger;

    public ChromaService(HttpClient httpClient, IConfiguration configuration, ILogger<ChromaService> logger)
    {
        _httpClient = httpClient;
        _configuration = configuration;
        _logger = logger;
    }

    public async Task<bool> TriggerReindexAsync(CancellationToken cancellationToken)
    {
        var baseUrl = _configuration["PythonRag:BaseUrl"] ?? "http://localhost:8000";
        var endpoint = $"{baseUrl.TrimEnd('/')}/reindex";

        _logger.LogInformation("Triggering Python RAG reindex via {Endpoint}.", endpoint);

        using var response = await _httpClient.PostAsync(endpoint, content: null, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            _logger.LogWarning("Python RAG reindex failed with {StatusCode}: {Body}", response.StatusCode, body);
            return false;
        }

        _logger.LogInformation("Python RAG reindex completed successfully.");
        return true;
    }
}
