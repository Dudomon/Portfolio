namespace BrazilianTaxLibrary.Models;

/// <summary>
/// Brazilian tax regimes
/// </summary>
public enum TaxRegime
{
    /// <summary>
    /// Normal regime (Lucro Real or Lucro Presumido)
    /// Full tax calculation with all federal, state, and municipal taxes
    /// </summary>
    NormalRegime,

    /// <summary>
    /// Simples Nacional - Simplified regime for small businesses
    /// Revenue up to R$ 4.8M/year
    /// Single monthly payment (DAS) covering multiple taxes
    /// </summary>
    SimplesNacional,

    /// <summary>
    /// MEI (Microempreendedor Individual)
    /// Revenue up to R$ 81K/year
    /// Fixed monthly payment
    /// </summary>
    MEI,

    /// <summary>
    /// Exempt entities (non-profit, government, etc.)
    /// </summary>
    Exempt,

    /// <summary>
    /// Special regime (ZFM - Zona Franca de Manaus, etc.)
    /// </summary>
    SpecialRegime
}
