namespace BrazilianTaxLibrary.Calculations;

using BrazilianTaxLibrary.Models;

/// <summary>
/// Calculator for Brazilian Tax Reform 2024 (LC 214/2024)
/// Implements IBS (Imposto sobre Bens e Serviços) and CBS (Contribuição sobre Bens e Serviços)
/// </summary>
public class TaxReformCalculator
{
    // Standard rates from LC 214/2024
    private const decimal IBS_STANDARD_RATE = 26.50m; // Expected final rate (to be confirmed)
    private const decimal CBS_STANDARD_RATE = 8.80m;  // Expected final rate (to be confirmed)

    /// <summary>
    /// Calculate IBS and CBS according to Tax Reform rules
    /// Implements transition period logic (2026-2033)
    /// </summary>
    public TaxReformCalculationResult CalculateIBS_CBS(TaxReformCalculationRequest request)
    {
        var result = new TaxReformCalculationResult
        {
            Year = request.Year
        };

        // Export operations are exempt
        if (request.IsExportOperation)
        {
            result.Notes = "Export operation - Exempt from IBS/CBS (Art. 5º, LC 214/2024)";
            return result;
        }

        // Determine transition percentage based on year
        result.TransitionPercentage = GetTransitionPercentage(request.Year);

        // Get applicable rates (may vary by NCM - essential goods have reduced rates)
        var rates = GetApplicableRates(request.NCM, request.ServiceCode);
        result.IBSRate = rates.IBSRate;
        result.CBSRate = rates.CBSRate;

        // Calculate new taxes (IBS/CBS) based on transition percentage
        result.IBS = CalculateIBS(request.BaseValue, rates.IBSRate, result.TransitionPercentage);
        result.CBS = CalculateCBS(request.BaseValue, rates.CBSRate, result.TransitionPercentage);

        // Calculate legacy taxes (ICMS, PIS, COFINS) - progressively reduced during transition
        var legacyPercentage = 100m - result.TransitionPercentage;
        result.LegacyICMS = CalculateLegacyICMS(request.BaseValue, request.OriginState, legacyPercentage);
        result.LegacyPIS = CalculateLegacyPIS(request.BaseValue, legacyPercentage);
        result.LegacyCOFINS = CalculateLegacyCOFINS(request.BaseValue, legacyPercentage);

        // Total tax burden
        result.TotalTax = result.IBS + result.CBS + result.LegacyICMS +
                          result.LegacyPIS + result.LegacyCOFINS;

        // Apply non-cumulative credit
        result.AvailableCredit = CalculateAvailableCredit(result.IBS, result.CBS, request.PreviousCredit);

        // Calculate cashback if eligible (low-income families)
        if (request.EligibleForCashback)
        {
            result.CashbackAmount = CalculateCashback(result.IBS, result.CBS, request.NCM);
        }

        // Net tax = Total - Credits - Cashback
        result.NetTax = result.TotalTax - result.AvailableCredit - result.CashbackAmount;

        // Calculation breakdown for audit
        result.CalculationBreakdown = new Dictionary<string, decimal>
        {
            ["Base Value"] = request.BaseValue,
            ["IBS"] = result.IBS,
            ["CBS"] = result.CBS,
            ["Legacy ICMS"] = result.LegacyICMS,
            ["Legacy PIS"] = result.LegacyPIS,
            ["Legacy COFINS"] = result.LegacyCOFINS,
            ["Total Before Credits"] = result.TotalTax,
            ["Available Credit"] = result.AvailableCredit,
            ["Cashback"] = result.CashbackAmount,
            ["Net Tax"] = result.NetTax
        };

        result.Notes = $"Transition year {request.Year} - {result.TransitionPercentage}% new system, " +
                       $"{legacyPercentage}% legacy system";

        return result;
    }

    /// <summary>
    /// Get transition percentage based on year
    /// 2026: 10%, 2027: 20%, ..., 2033: 100%
    /// </summary>
    private decimal GetTransitionPercentage(int year)
    {
        return year switch
        {
            <= 2025 => 0m,     // Pre-reform
            2026 => 10m,       // Year 1
            2027 => 20m,       // Year 2
            2028 => 30m,       // Year 3
            2029 => 40m,       // Year 4
            2030 => 50m,       // Year 5
            2031 => 70m,       // Year 6
            2032 => 90m,       // Year 7
            >= 2033 => 100m    // Full implementation
        };
    }

    /// <summary>
    /// Get applicable IBS/CBS rates based on product/service classification
    /// Essential goods have reduced rates
    /// </summary>
    private (decimal IBSRate, decimal CBSRate) GetApplicableRates(string ncm, string? serviceCode)
    {
        // Essential goods (food, medicine, etc.) - 40% reduction
        if (IsEssentialGood(ncm))
        {
            return (IBS_STANDARD_RATE * 0.60m, CBS_STANDARD_RATE * 0.60m);
        }

        // Health services - 60% reduction
        if (IsHealthService(serviceCode))
        {
            return (IBS_STANDARD_RATE * 0.40m, CBS_STANDARD_RATE * 0.40m);
        }

        // Education services - 70% reduction
        if (IsEducationService(serviceCode))
        {
            return (IBS_STANDARD_RATE * 0.30m, CBS_STANDARD_RATE * 0.30m);
        }

        // Standard rate
        return (IBS_STANDARD_RATE, CBS_STANDARD_RATE);
    }

    private decimal CalculateIBS(decimal baseValue, decimal rate, decimal transitionPercentage)
    {
        var fullTax = baseValue * (rate / 100m);
        return fullTax * (transitionPercentage / 100m);
    }

    private decimal CalculateCBS(decimal baseValue, decimal rate, decimal transitionPercentage)
    {
        var fullTax = baseValue * (rate / 100m);
        return fullTax * (transitionPercentage / 100m);
    }

    private decimal CalculateLegacyICMS(decimal baseValue, FederalUnit state, decimal legacyPercentage)
    {
        // Typical ICMS rate (varies by state, using 18% as example)
        var icmsRate = GetICMSRateByState(state);
        var fullICMS = baseValue * (icmsRate / 100m);
        return fullICMS * (legacyPercentage / 100m);
    }

    private decimal CalculateLegacyPIS(decimal baseValue, decimal legacyPercentage)
    {
        // Standard PIS rate: 1.65% (non-cumulative regime)
        var fullPIS = baseValue * 0.0165m;
        return fullPIS * (legacyPercentage / 100m);
    }

    private decimal CalculateLegacyCOFINS(decimal baseValue, decimal legacyPercentage)
    {
        // Standard COFINS rate: 7.6% (non-cumulative regime)
        var fullCOFINS = baseValue * 0.076m;
        return fullCOFINS * (legacyPercentage / 100m);
    }

    private decimal CalculateAvailableCredit(decimal ibs, decimal cbs, decimal previousCredit)
    {
        // IBS and CBS are non-cumulative - credit on previous stage
        return previousCredit + (ibs + cbs);
    }

    private decimal CalculateCashback(decimal ibs, decimal cbs, string ncm)
    {
        // Cashback for essential goods consumed by low-income families
        // Typically 20% of IBS + 50% of CBS on essential goods
        if (IsEssentialGood(ncm))
        {
            return (ibs * 0.20m) + (cbs * 0.50m);
        }
        return 0m;
    }

    private decimal GetICMSRateByState(FederalUnit state)
    {
        // Simplified - actual rates vary by state and product
        return state switch
        {
            FederalUnit.SP => 18m,
            FederalUnit.RJ => 20m,
            FederalUnit.MG => 18m,
            FederalUnit.RS => 17m,
            FederalUnit.PR => 18m,
            FederalUnit.SC => 17m,
            _ => 18m // Default
        };
    }

    private bool IsEssentialGood(string ncm)
    {
        // Simplified check - real implementation would have full NCM table
        // Essential goods: food, medicine, hygiene products
        var essentialPrefixes = new[] { "02", "04", "10", "11", "30", "33" };
        return essentialPrefixes.Any(prefix => ncm.StartsWith(prefix));
    }

    private bool IsHealthService(string? serviceCode)
    {
        // LC 116/2003 - Service list codes for health
        if (string.IsNullOrEmpty(serviceCode)) return false;
        return serviceCode.StartsWith("04"); // Health services
    }

    private bool IsEducationService(string? serviceCode)
    {
        // LC 116/2003 - Service list codes for education
        if (string.IsNullOrEmpty(serviceCode)) return false;
        return serviceCode.StartsWith("08"); // Education services
    }
}
