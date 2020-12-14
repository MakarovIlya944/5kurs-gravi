namespace MKE.Problem.InverseProblem
{
    public class FittingConfiguration
    {
        public double MinValue { get; set; } = 0.01d;

        public double MaxPercentIncrease { get; set; } = 1;

        public double MaxPercentDecrease { get; set; } = 1;

        public double MaxValue { get; set; } = 100d;

        public double StartedValue { get; set; } = 0.01;

        public double Step { get; set; }
    }
}