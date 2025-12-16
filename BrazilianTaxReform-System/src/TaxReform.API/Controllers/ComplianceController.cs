using Microsoft.AspNetCore.Mvc;
using TaxReform.Core.Compliance;
using TaxReform.Core.Models;

namespace TaxReform.API.Controllers;

/// <summary>
/// Compliance and gap analysis endpoints for Tax Reform readiness
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class ComplianceController : ControllerBase
{
    private readonly IComplianceService _complianceService;
    private readonly ILogger<ComplianceController> _logger;

    public ComplianceController(
        IComplianceService complianceService,
        ILogger<ComplianceController> logger)
    {
        _complianceService = complianceService;
        _logger = logger;
    }

    /// <summary>
    /// Perform gap analysis to assess system readiness for Tax Reform
    /// </summary>
    /// <param name="request">System capabilities and target year</param>
    /// <returns>Detailed gap analysis with critical, high, and medium gaps</returns>
    /// <response code="200">Analysis completed successfully</response>
    [HttpPost("gap-analysis")]
    [ProducesResponseType(typeof(GapAnalysisResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<GapAnalysisResult>> PerformGapAnalysis(
        [FromBody] GapAnalysisRequest request)
    {
        _logger.LogInformation(
            "Performing gap analysis for system version {Version}, target year {Year}",
            request.SystemVersion, request.TargetYear);

        var result = await _complianceService.AnalyzeGapsAsync(request);

        _logger.LogInformation(
            "Gap analysis completed. Readiness score: {Score}, Status: {Status}",
            result.ReadinessScore, result.OverallStatus);

        return Ok(result);
    }

    /// <summary>
    /// Get real-time compliance status dashboard
    /// </summary>
    /// <returns>Compliance dashboard with metrics and alerts</returns>
    [HttpGet("dashboard")]
    [ProducesResponseType(typeof(ComplianceDashboard), StatusCodes.Status200OK)]
    public async Task<ActionResult<ComplianceDashboard>> GetComplianceDashboard()
    {
        var dashboard = await _complianceService.GetDashboardAsync();
        return Ok(dashboard);
    }

    /// <summary>
    /// Validate compliance with specific article of LC 214/2024
    /// </summary>
    /// <param name="articleNumber">Article number to validate (e.g., "3" for Art. 3º)</param>
    /// <returns>Compliance status for the specified article</returns>
    [HttpGet("validate-article/{articleNumber}")]
    [ProducesResponseType(typeof(ArticleComplianceResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<ArticleComplianceResult>> ValidateArticle(
        string articleNumber)
    {
        var result = await _complianceService.ValidateArticleComplianceAsync(articleNumber);
        return Ok(result);
    }

    /// <summary>
    /// Get list of all legal requirements from LC 214/2024
    /// </summary>
    /// <returns>Complete list of legal requirements with implementation status</returns>
    [HttpGet("legal-requirements")]
    [ProducesResponseType(typeof(List<LegalRequirement>), StatusCodes.Status200OK)]
    public async Task<ActionResult<List<LegalRequirement>>> GetLegalRequirements()
    {
        var requirements = await _complianceService.GetAllLegalRequirementsAsync();
        return Ok(requirements);
    }

    /// <summary>
    /// Generate compliance report for audit purposes
    /// </summary>
    /// <param name="request">Report generation parameters</param>
    /// <returns>PDF report download</returns>
    [HttpPost("generate-report")]
    [ProducesResponseType(typeof(FileResult), StatusCodes.Status200OK)]
    public async Task<IActionResult> GenerateComplianceReport(
        [FromBody] ComplianceReportRequest request)
    {
        _logger.LogInformation("Generating compliance report for period {StartDate} to {EndDate}",
            request.StartDate, request.EndDate);

        var pdfBytes = await _complianceService.GenerateReportAsync(request);

        return File(pdfBytes, "application/pdf", $"compliance-report-{DateTime.UtcNow:yyyyMMdd}.pdf");
    }

    /// <summary>
    /// Get recommended actions based on gap analysis
    /// </summary>
    /// <param name="targetYear">Target implementation year</param>
    /// <returns>Prioritized list of recommended actions</returns>
    [HttpGet("recommended-actions")]
    [ProducesResponseType(typeof(List<RecommendedAction>), StatusCodes.Status200OK)]
    public async Task<ActionResult<List<RecommendedAction>>> GetRecommendedActions(
        [FromQuery] int targetYear = 2027)
    {
        var actions = await _complianceService.GetRecommendedActionsAsync(targetYear);
        return Ok(actions);
    }

    /// <summary>
    /// Check go-live readiness for Tax Reform implementation
    /// </summary>
    /// <param name="goLiveDate">Planned go-live date</param>
    /// <returns>Go-live readiness assessment</returns>
    [HttpGet("go-live-readiness")]
    [ProducesResponseType(typeof(GoLiveReadiness), StatusCodes.Status200OK)]
    public async Task<ActionResult<GoLiveReadiness>> CheckGoLiveReadiness(
        [FromQuery] DateTime goLiveDate)
    {
        _logger.LogInformation("Checking go-live readiness for date {GoLiveDate}", goLiveDate);

        var readiness = await _complianceService.AssessGoLiveReadinessAsync(goLiveDate);

        return Ok(readiness);
    }
}
