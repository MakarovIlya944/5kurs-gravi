using System;
using System.Collections.Generic;

namespace MKE.Utils {
    public class DisposableContainer : IDisposable
    {
        private readonly IEnumerable<IDisposable> _array;

        public DisposableContainer(IEnumerable<IDisposable> array)
        {
            _array = array;
        }

        public void Dispose()
        {
            foreach (var disposable in _array)
            {
                disposable.Dispose();
            }
        }
    }
}