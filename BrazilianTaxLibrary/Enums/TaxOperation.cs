namespace BrazilianTaxLibrary.Models;

/// <summary>
/// Types of fiscal operations
/// </summary>
public enum TaxOperation
{
    /// <summary>Sale within same state</summary>
    IntrastateSale,

    /// <summary>Sale between different states</summary>
    Interstate,

    /// <summary>Import from foreign country</summary>
    Import,

    /// <summary>Export to foreign country</summary>
    Export,

    /// <summary>Return of goods</summary>
    Return,

    /// <summary>Transfer between company locations</summary>
    Transfer,

    /// <summary>Service provision</summary>
    Service,

    /// <summary>Manufacturing operation</summary>
    Manufacturing
}
