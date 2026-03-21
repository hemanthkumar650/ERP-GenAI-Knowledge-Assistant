using System.Security.Cryptography;
using System.Text;

namespace dotnet_worker.Services;

public class PoliciesService
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<PoliciesService> _logger;

    public PoliciesService(IConfiguration configuration, ILogger<PoliciesService> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    public string GetPoliciesPath()
    {
        return _configuration["PoliciesPath"] ?? "../data/policies";
    }

    public string BuildSnapshot()
    {
        var policiesPath = GetPoliciesPath();
        Directory.CreateDirectory(policiesPath);

        var files = Directory
            .EnumerateFiles(policiesPath, "*.pdf", SearchOption.TopDirectoryOnly)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .Select(path =>
            {
                var fileInfo = new FileInfo(path);
                return $"{fileInfo.Name}|{fileInfo.Length}|{fileInfo.LastWriteTimeUtc.Ticks}";
            });

        var payload = string.Join("\n", files);
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(payload));
        var snapshot = Convert.ToHexString(bytes);

        _logger.LogInformation("Policy snapshot computed for {Path}.", policiesPath);
        return snapshot;
    }
}
