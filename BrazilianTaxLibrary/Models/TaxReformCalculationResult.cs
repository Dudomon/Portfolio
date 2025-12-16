namespace BrazilianTaxLibrary.Models;

/// <summary>
/// Result of Tax Reform calculation with IBS/CBS and legacy tax breakdown
/// </summary>
public class TaxReformCalculationResult
{
    /// <summary>
    /// IBS (Imposto sobre Bens e Serviços) - Replaces ICMS + ISS
    /// </summary>
    public decimal IBS { get; set; }

    /// <summary>
    /// CBS (Contribuição sobre Bens e Serviços) - Replaces PIS + COFINS
    /// </summary>
    public decimal CBS { get; set; }

    /// <summary>
    /// Legacy ICMS (during transition period 2026-2033)
    /// </summary>
    public decimal LegacyICMS { get; set; }

    /// <summary>
    /// Legacy PIS (during transition period 2026-2033)
    /// </summary>
    public decimal LegacyPIS { get; set; }

    /// <summary>
    /// Legacy COFINS (during transition period 2026-2033)
    /// </summary>
    public decimal LegacyCOFINS { get; set; }

    /// <summary>
    /// Total tax burden (IBS + CBS + legacy taxes during transition)
    /// </summary>
    public decimal TotalTax { get; set; }

    /// <summary>
    /// Cashback amount (if eligible)
    /// </summary>
    public decimal CashbackAmount { get; set; }

    /// <summary>
    /// Available credit from chain (non-cumulative)
    /// </summary>
    public decimal AvailableCredit { get; set; }

    /// <summary>
    /// Net tax after credits and cashback
    /// </summary>
    public decimal NetTax { get; set; }

    /// <summary>
    /// Transition year information
    /// </summary>
    public int Year { get; set; }

    /// <summary>
    /// IBS rate applied (percentage)
    /// </summary>
    public decimal IBSRate { get; set; }

    /// <summary>
    /// CBS rate applied (percentage)
    /// </summary>
    public decimal CBSRate { get; set; }

    /// <summary>
    /// Transition percentage (0-100%)
    /// 2026: 10%, 2027: 20%, ..., 2033: 100%
    /// </summary>
    public decimal TransitionPercentage { get; set; }

    /// <summary>
    /// Detailed calculation breakdown
    /// </summary>
    public Dictionary<string, decimal> CalculationBreakdown { get; set; } = new();

    /// <summary>
    /// Notes and observations
    /// </summary>
    public string? Notes { get; set; }
}
