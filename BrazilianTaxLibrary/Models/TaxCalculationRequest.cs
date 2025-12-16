namespace BrazilianTaxLibrary.Models;

/// <summary>
/// Base request model for tax calculations
/// </summary>
public class TaxCalculationRequest
{
    /// <summary>
    /// Base value for tax calculation (product/service value)
    /// </summary>
    public decimal BaseValue { get; set; }

    /// <summary>
    /// Tax rate percentage (e.g., 18.00 for 18%)
    /// </summary>
    public decimal TaxRate { get; set; }

    /// <summary>
    /// Origin state (UF)
    /// </summary>
    public FederalUnit OriginState { get; set; }

    /// <summary>
    /// Destination state (UF) for interstate operations
    /// </summary>
    public FederalUnit? DestinationState { get; set; }

    /// <summary>
    /// Type of fiscal operation
    /// </summary>
    public TaxOperation OperationType { get; set; }

    /// <summary>
    /// NCM (Nomenclatura Comum do Mercosul) code
    /// </summary>
    public string? NCM { get; set; }

    /// <summary>
    /// CFOP (Código Fiscal de Operações e Prestações)
    /// </summary>
    public string? CFOP { get; set; }

    /// <summary>
    /// Customer tax regime
    /// </summary>
    public TaxRegime CustomerRegime { get; set; } = TaxRegime.NormalRegime;

    /// <summary>
    /// Whether the operation is an export
    /// </summary>
    public bool IsExportOperation { get; set; } = false;

    /// <summary>
    /// Whether the customer is final consumer
    /// </summary>
    public bool IsFinalConsumer { get; set; } = false;
}
