namespace BrazilianTaxLibrary.Models;

/// <summary>
/// Result of tax calculation with detailed breakdown
/// </summary>
public class TaxCalculationResult
{
    /// <summary>
    /// Calculated tax value
    /// </summary>
    public decimal TaxValue { get; set; }

    /// <summary>
    /// Base value used for calculation
    /// </summary>
    public decimal BaseValue { get; set; }

    /// <summary>
    /// Applied tax rate percentage
    /// </summary>
    public decimal AppliedRate { get; set; }

    /// <summary>
    /// Tax type (ICMS, IPI, PIS, COFINS, etc.)
    /// </summary>
    public string TaxType { get; set; } = string.Empty;

    /// <summary>
    /// CST (Código de Situação Tributária) applied
    /// </summary>
    public string? CST { get; set; }

    /// <summary>
    /// Additional information or observations
    /// </summary>
    public string? Notes { get; set; }

    /// <summary>
    /// Whether tax is recoverable as credit
    /// </summary>
    public bool IsRecoverable { get; set; } = false;

    /// <summary>
    /// Calculation breakdown for audit trail
    /// </summary>
    public Dictionary<string, decimal> Breakdown { get; set; } = new();
}
