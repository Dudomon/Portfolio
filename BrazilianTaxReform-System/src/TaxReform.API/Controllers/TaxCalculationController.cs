using Microsoft.AspNetCore.Mvc;
using TaxReform.Core.Services;
using TaxReform.Core.Models;

namespace TaxReform.API.Controllers;

/// <summary>
/// Tax calculation endpoints for Brazilian Tax Reform (IBS/CBS)
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class TaxCalculationController : ControllerBase
{
    private readonly ITaxCalculationService _taxService;
    private readonly ILogger<TaxCalculationController> _logger;

    public TaxCalculationController(
        ITaxCalculationService taxService,
        ILogger<TaxCalculationController> logger)
    {
        _taxService = taxService;
        _logger = logger;
    }

    /// <summary>
    /// Calculate IBS and CBS according to Tax Reform rules with transition period
    /// </summary>
    /// <param name="request">Tax calculation parameters</param>
    /// <returns>Detailed tax calculation result with IBS, CBS, and legacy taxes</returns>
    /// <response code="200">Calculation successful</response>
    /// <response code="400">Invalid request parameters</response>
    [HttpPost("calculate-reform")]
    [ProducesResponseType(typeof(TaxReformCalculationResult), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ProblemDetails), StatusCodes.Status400BadRequest)]
    public async Task<ActionResult<TaxReformCalculationResult>> CalculateReform(
        [FromBody] TaxReformCalculationRequest request)
    {
        try
        {
            _logger.LogInformation(
                "Calculating tax reform for base value {BaseValue}, year {Year}, NCM {NCM}",
                request.BaseValue, request.Year, request.NCM);

            var result = await _taxService.CalculateTaxReformAsync(request);

            _logger.LogInformation(
                "Tax calculation completed. IBS: {IBS}, CBS: {CBS}, Total: {Total}",
                result.IBS, result.CBS, result.TotalTax);

            return Ok(result);
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning(ex, "Invalid tax calculation request");
            return BadRequest(new ProblemDetails
            {
                Title = "Invalid Request",
                Detail = ex.Message,
                Status = StatusCodes.Status400BadRequest
            });
        }
    }

    /// <summary>
    /// Calculate legacy taxes (ICMS, PIS, COFINS) - pre-reform system
    /// </summary>
    /// <param name="request">Legacy tax calculation parameters</param>
    /// <returns>Legacy tax calculation result</returns>
    [HttpPost("calculate-legacy")]
    [ProducesResponseType(typeof(LegacyTaxResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<LegacyTaxResult>> CalculateLegacy(
        [FromBody] LegacyTaxRequest request)
    {
        var result = await _taxService.CalculateLegacyTaxesAsync(request);
        return Ok(result);
    }

    /// <summary>
    /// Calculate cashback amount for low-income families
    /// </summary>
    /// <param name="request">Cashback calculation parameters</param>
    /// <returns>Cashback amount and eligibility details</returns>
    [HttpPost("calculate-cashback")]
    [ProducesResponseType(typeof(CashbackResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<CashbackResult>> CalculateCashback(
        [FromBody] CashbackRequest request)
    {
        var result = await _taxService.CalculateCashbackAsync(request);
        return Ok(result);
    }

    /// <summary>
    /// Get applicable tax rates for a specific product/service
    /// </summary>
    /// <param name="ncm">NCM code (optional)</param>
    /// <param name="serviceCode">Service code (optional)</param>
    /// <param name="year">Target year for rates</param>
    /// <returns>Applicable IBS and CBS rates</returns>
    [HttpGet("rates")]
    [ProducesResponseType(typeof(TaxRatesResponse), StatusCodes.Status200OK)]
    public async Task<ActionResult<TaxRatesResponse>> GetApplicableRates(
        [FromQuery] string? ncm = null,
        [FromQuery] string? serviceCode = null,
        [FromQuery] int year = 2027)
    {
        var rates = await _taxService.GetApplicableRatesAsync(ncm, serviceCode, year);
        return Ok(rates);
    }

    /// <summary>
    /// Validate if an operation is exempt from IBS/CBS
    /// </summary>
    /// <param name="request">Exemption validation parameters</param>
    /// <returns>Exemption status and legal basis</returns>
    [HttpPost("validate-exemption")]
    [ProducesResponseType(typeof(ExemptionResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<ExemptionResult>> ValidateExemption(
        [FromBody] ExemptionRequest request)
    {
        var result = await _taxService.ValidateExemptionAsync(request);
        return Ok(result);
    }

    /// <summary>
    /// Calculate tax credit available in the non-cumulative chain
    /// </summary>
    /// <param name="request">Credit calculation parameters</param>
    /// <returns>Available credit amount</returns>
    [HttpPost("calculate-credit")]
    [ProducesResponseType(typeof(CreditResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<CreditResult>> CalculateCredit(
        [FromBody] CreditRequest request)
    {
        var result = await _taxService.CalculateCreditAsync(request);
        return Ok(result);
    }

    /// <summary>
    /// Batch calculation for multiple items (e.g., invoice with multiple line items)
    /// </summary>
    /// <param name="request">Batch calculation request with multiple items</param>
    /// <returns>Batch calculation results</returns>
    [HttpPost("calculate-batch")]
    [ProducesResponseType(typeof(BatchCalculationResult), StatusCodes.Status200OK)]
    public async Task<ActionResult<BatchCalculationResult>> CalculateBatch(
        [FromBody] BatchCalculationRequest request)
    {
        _logger.LogInformation(
            "Processing batch calculation with {ItemCount} items",
            request.Items.Count);

        var result = await _taxService.CalculateBatchAsync(request);

        _logger.LogInformation(
            "Batch calculation completed. Total IBS: {IBS}, Total CBS: {CBS}",
            result.TotalIBS, result.TotalCBS);

        return Ok(result);
    }
}
