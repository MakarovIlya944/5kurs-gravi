using System;

namespace MKE.Utils
{
    public class DisposableChanges : IDisposable
    {
        private readonly Action _dispose;

        public DisposableChanges(Action dispose)
        {
            _dispose = dispose;
        }

        public void Dispose()
        {
            _dispose();
        }
    }
}