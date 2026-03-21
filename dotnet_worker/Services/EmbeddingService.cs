namespace dotnet_worker.Services;

public class EmbeddingService
{
    private readonly ChromaService _chromaService;
    private readonly ILogger<EmbeddingService> _logger;

    public EmbeddingService(ChromaService chromaService, ILogger<EmbeddingService> logger)
    {
        _chromaService = chromaService;
        _logger = logger;
    }

    public async Task<bool> RefreshEmbeddingsAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Refreshing embeddings and vector index.");
        return await _chromaService.TriggerReindexAsync(cancellationToken);
    }
}
