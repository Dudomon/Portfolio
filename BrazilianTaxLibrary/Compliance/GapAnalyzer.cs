namespace BrazilianTaxLibrary.Compliance;

/// <summary>
/// Gap Analysis tool for Tax Reform readiness assessment
/// Compares current system capabilities against legal requirements
/// </summary>
public class GapAnalyzer
{
    /// <summary>
    /// Analyze system readiness for Tax Reform implementation
    /// </summary>
    public GapAnalysisResult AnalyzeTaxReformReadiness(
        SystemCapabilities currentSystem,
        int targetYear)
    {
        var result = new GapAnalysisResult
        {
            AnalysisDate = DateTime.UtcNow,
            TargetYear = targetYear,
            SystemVersion = currentSystem.Version
        };

        // Check IBS calculation capability
        if (!currentSystem.SupportsIBSCalculation)
        {
            result.CriticalGaps.Add(new Gap
            {
                Requirement = "IBS (Imposto sobre Bens e Serviços) calculation engine",
                Severity = GapSeverity.Critical,
                LegalReference = "LC 214/2024, Art. 3º",
                CurrentStatus = "Not Implemented",
                RemediationAction = "Implement IBS calculation with transition rules (2026-2033)",
                EstimatedEffort = "High - 3-4 weeks",
                Priority = 1
            });
        }

        // Check CBS calculation capability
        if (!currentSystem.SupportsCBSCalculation)
        {
            result.CriticalGaps.Add(new Gap
            {
                Requirement = "CBS (Contribuição sobre Bens e Serviços) calculation engine",
                Severity = GapSeverity.Critical,
                LegalReference = "LC 214/2024, Art. 4º",
                CurrentStatus = "Not Implemented",
                RemediationAction = "Implement CBS calculation with non-cumulative credit",
                EstimatedEffort = "High - 3-4 weeks",
                Priority = 1
            });
        }

        // Check transition period handling
        if (!currentSystem.SupportsTransitionPeriod)
        {
            result.CriticalGaps.Add(new Gap
            {
                Requirement = "Transition period tax calculation (2026-2033)",
                Severity = GapSeverity.Critical,
                LegalReference = "LC 214/2024, Art. 25º (Disposições Transitórias)",
                CurrentStatus = "Not Implemented",
                RemediationAction = "Implement dual calculation: IBS/CBS + legacy ICMS/PIS/COFINS with progressive rates",
                EstimatedEffort = "Medium - 2 weeks",
                Priority = 2
            });
        }

        // Check cashback system
        if (!currentSystem.SupportsCashback)
        {
            result.HighGaps.Add(new Gap
            {
                Requirement = "Cashback system for low-income families",
                Severity = GapSeverity.High,
                LegalReference = "LC 214/2024, Art. 15º",
                CurrentStatus = "Not Implemented",
                RemediationAction = "Implement cashback calculation based on family income and essential goods consumption",
                EstimatedEffort = "Medium - 2 weeks",
                Priority = 3
            });
        }

        // Check split payment capability
        if (!currentSystem.SupportsSplitPayment)
        {
            result.HighGaps.Add(new Gap
            {
                Requirement = "Split payment (pagamento fracionado)",
                Severity = GapSeverity.High,
                LegalReference = "LC 214/2024, Art. 28º",
                CurrentStatus = "Not Implemented",
                RemediationAction = "Implement split payment mechanism for tax collection at transaction time",
                EstimatedEffort = "High - 3 weeks",
                Priority = 3
            });
        }

        // Check tax credit chain tracking
        if (!currentSystem.SupportsNonCumulativeCredit)
        {
            result.HighGaps.Add(new Gap
            {
                Requirement = "Non-cumulative tax credit chain tracking",
                Severity = GapSeverity.High,
                LegalReference = "LC 214/2024, Art. 10º",
                CurrentStatus = "Partial",
                RemediationAction = "Extend credit tracking to include IBS/CBS throughout the supply chain",
                EstimatedEffort = "Medium - 2 weeks",
                Priority = 4
            });
        }

        // Check special regimes support
        if (!currentSystem.SupportsSpecialRegimes)
        {
            result.MediumGaps.Add(new Gap
            {
                Requirement = "Special regimes (ZFM, Simples Nacional adaptation)",
                Severity = GapSeverity.Medium,
                LegalReference = "LC 214/2024, Art. 20º-23º",
                CurrentStatus = "Not Implemented",
                RemediationAction = "Implement special regime rules for ZFM and adapt Simples Nacional calculations",
                EstimatedEffort = "Medium - 2 weeks",
                Priority = 5
            });
        }

        // Check master data readiness
        if (!currentSystem.HasUpdatedNCMTable)
        {
            result.MediumGaps.Add(new Gap
            {
                Requirement = "Updated NCM table with IBS/CBS rates by product category",
                Severity = GapSeverity.Medium,
                LegalReference = "To be published by Federal Revenue",
                CurrentStatus = "Outdated",
                RemediationAction = "Update NCM master data with new tax rates for essential goods, health, education",
                EstimatedEffort = "Low - 1 week",
                Priority = 6
            });
        }

        // Check reporting capability
        if (!currentSystem.SupportsNewFiscalDocuments)
        {
            result.MediumGaps.Add(new Gap
            {
                Requirement = "New fiscal document layouts (NF-e with IBS/CBS fields)",
                Severity = GapSeverity.Medium,
                LegalReference = "Nota Técnica (to be published)",
                CurrentStatus = "Not Ready",
                RemediationAction = "Update NF-e/CT-e XML generation to include IBS/CBS tax breakdown",
                EstimatedEffort = "Medium - 2 weeks",
                Priority = 7
            });
        }

        // Calculate readiness score
        result.ReadinessScore = CalculateReadinessScore(result);
        result.OverallStatus = DetermineOverallStatus(result.ReadinessScore, targetYear);

        return result;
    }

    private int CalculateReadinessScore(GapAnalysisResult result)
    {
        var maxScore = 100;
        var deductions =
            (result.CriticalGaps.Count * 20) +
            (result.HighGaps.Count * 10) +
            (result.MediumGaps.Count * 5);

        return Math.Max(0, maxScore - deductions);
    }

    private string DetermineOverallStatus(int readinessScore, int targetYear)
    {
        var yearsUntilTarget = targetYear - DateTime.UtcNow.Year;

        if (readinessScore >= 90)
            return "Ready";

        if (readinessScore >= 70 && yearsUntilTarget >= 1)
            return "On Track";

        if (readinessScore >= 50 && yearsUntilTarget >= 2)
            return "Needs Attention";

        return "Critical - Immediate Action Required";
    }
}

public class SystemCapabilities
{
    public string Version { get; set; } = "1.0";
    public bool SupportsIBSCalculation { get; set; }
    public bool SupportsCBSCalculation { get; set; }
    public bool SupportsTransitionPeriod { get; set; }
    public bool SupportsCashback { get; set; }
    public bool SupportsSplitPayment { get; set; }
    public bool SupportsNonCumulativeCredit { get; set; }
    public bool SupportsSpecialRegimes { get; set; }
    public bool HasUpdatedNCMTable { get; set; }
    public bool SupportsNewFiscalDocuments { get; set; }

    public static SystemCapabilities FromConfig()
    {
        // Load from configuration - simplified example
        return new SystemCapabilities();
    }
}

public class GapAnalysisResult
{
    public DateTime AnalysisDate { get; set; }
    public int TargetYear { get; set; }
    public string SystemVersion { get; set; } = string.Empty;
    public List<Gap> CriticalGaps { get; set; } = new();
    public List<Gap> HighGaps { get; set; } = new();
    public List<Gap> MediumGaps { get; set; } = new();
    public int ReadinessScore { get; set; }
    public string OverallStatus { get; set; } = string.Empty;
}

public class Gap
{
    public string Requirement { get; set; } = string.Empty;
    public GapSeverity Severity { get; set; }
    public string LegalReference { get; set; } = string.Empty;
    public string CurrentStatus { get; set; } = string.Empty;
    public string RemediationAction { get; set; } = string.Empty;
    public string EstimatedEffort { get; set; } = string.Empty;
    public int Priority { get; set; }
}

public enum GapSeverity
{
    Critical,
    High,
    Medium,
    Low
}
