namespace BrazilianTaxLibrary.Validators;

/// <summary>
/// CNPJ (Cadastro Nacional da Pessoa Jurídica) validator
/// Brazilian company tax identification number
/// Format: XX.XXX.XXX/XXXX-XX (14 digits)
/// </summary>
public class CNPJValidator
{
    /// <summary>
    /// Validate CNPJ using check digit algorithm
    /// </summary>
    /// <param name="cnpj">CNPJ string (with or without formatting)</param>
    /// <returns>True if valid, false otherwise</returns>
    public bool Validate(string cnpj)
    {
        if (string.IsNullOrWhiteSpace(cnpj))
            return false;

        // Remove formatting
        cnpj = new string(cnpj.Where(char.IsDigit).ToArray());

        // Must have exactly 14 digits
        if (cnpj.Length != 14)
            return false;

        // Check for known invalid patterns (all same digits)
        if (cnpj.Distinct().Count() == 1)
            return false;

        // Validate first check digit
        var firstCheckDigit = CalculateCheckDigit(cnpj.Substring(0, 12), new[] { 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2 });
        if (cnpj[12] != firstCheckDigit)
            return false;

        // Validate second check digit
        var secondCheckDigit = CalculateCheckDigit(cnpj.Substring(0, 13), new[] { 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2 });
        if (cnpj[13] != secondCheckDigit)
            return false;

        return true;
    }

    /// <summary>
    /// Format CNPJ with standard mask XX.XXX.XXX/XXXX-XX
    /// </summary>
    public string Format(string cnpj)
    {
        cnpj = new string(cnpj.Where(char.IsDigit).ToArray());

        if (cnpj.Length != 14)
            throw new ArgumentException("CNPJ must have 14 digits");

        return $"{cnpj.Substring(0, 2)}.{cnpj.Substring(2, 3)}.{cnpj.Substring(5, 3)}/" +
               $"{cnpj.Substring(8, 4)}-{cnpj.Substring(12, 2)}";
    }

    private char CalculateCheckDigit(string cnpjPart, int[] weights)
    {
        var sum = 0;
        for (int i = 0; i < cnpjPart.Length; i++)
        {
            sum += int.Parse(cnpjPart[i].ToString()) * weights[i];
        }

        var remainder = sum % 11;
        var checkDigit = remainder < 2 ? 0 : 11 - remainder;

        return checkDigit.ToString()[0];
    }
}
