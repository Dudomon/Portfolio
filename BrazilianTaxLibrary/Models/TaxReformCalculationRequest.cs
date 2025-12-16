namespace BrazilianTaxLibrary.Models;

/// <summary>
/// Request model for Tax Reform calculations (IBS/CBS)
/// LC 214/2024 - Brazilian Tax Reform 2024
/// </summary>
public class TaxReformCalculationRequest
{
    /// <summary>
    /// Base value for calculation
    /// </summary>
    public decimal BaseValue { get; set; }

    /// <summary>
    /// Year of the operation (affects transition rates)
    /// 2026-2033: Transition period with progressive rates
    /// 2033+: Full implementation
    /// </summary>
    public int Year { get; set; }

    /// <summary>
    /// NCM (Nomenclatura Comum do Mercosul) code for product classification
    /// </summary>
    public string NCM { get; set; } = string.Empty;

    /// <summary>
    /// Origin state for IBS calculation
    /// </summary>
    public FederalUnit OriginState { get; set; }

    /// <summary>
    /// Destination state for IBS calculation
    /// </summary>
    public FederalUnit? DestinationState { get; set; }

    /// <summary>
    /// Whether the operation is an export (exempt from IBS/CBS)
    /// </summary>
    public bool IsExportOperation { get; set; } = false;

    /// <summary>
    /// Whether cashback applies (low-income families)
    /// </summary>
    public bool EligibleForCashback { get; set; } = false;

    /// <summary>
    /// Previous credit available in the chain (non-cumulative)
    /// </summary>
    public decimal PreviousCredit { get; set; } = 0m;

    /// <summary>
    /// Special regime flag (Simples Nacional, ZFM, etc.)
    /// </summary>
    public bool IsSpecialRegime { get; set; } = false;

    /// <summary>
    /// Service type for ISS/IBS calculation (if applicable)
    /// </summary>
    public string? ServiceCode { get; set; }
}
