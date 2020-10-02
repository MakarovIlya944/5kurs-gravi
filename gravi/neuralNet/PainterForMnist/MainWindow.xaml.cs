using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;

namespace PainterForMnist
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private static string modelPath = @"";

        public MainWindow()
        {
            InitializeComponent();
            BrushColorCombo.ItemsSource = typeof(Colors).GetProperties();
            var colors = BrushColorCombo.ItemsSource.Cast<PropertyInfo>().ToArray();
            for (var i = 0; i < colors.Length; i++)
            {
                if (colors[i].Name != "Black")
                {
                    continue;
                }

                BrushColorCombo.SelectedIndex = i;
                break;
            }
        }

        public void SaveToFilePng(Uri path, InkCanvas surface)
        {
            if (path == null) return;
            var bitmapGreyscale = ConvertToGrayScaleBitmap(surface);
            using (var outStream = new FileStream(path.LocalPath, FileMode.Create))
            {
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bitmapGreyscale));
                encoder.Save(outStream);
            }
        }

        public byte[] GetImageLikeByteArray(InkCanvas surface)
        {
            var bitmapGreyscale = ConvertToGrayScaleBitmap(surface);
            var bytesPerPixel = (bitmapGreyscale.Format.BitsPerPixel + 7) / 8;
            var stride = bitmapGreyscale.PixelWidth * bytesPerPixel;
            var bufferSize = bitmapGreyscale.PixelHeight * stride;
            var buffer = new byte[bufferSize];
            bitmapGreyscale.CopyPixels(buffer, stride, 0);
            return buffer;
        }

        private static FormatConvertedBitmap ConvertToGrayScaleBitmap(InkCanvas surface)
        {
            var transform = surface.LayoutTransform;
            surface.LayoutTransform = null;
            var size = new Size(surface.Width, surface.Height);
            surface.Measure(size);
            surface.Arrange(new Rect(size));
            var renderBitmap = new RenderTargetBitmap(28, 28, 96d, 96d, PixelFormats.Pbgra32);
            var visual = new DrawingVisual();
            using (var context = visual.RenderOpen())
            {
                var brush = new VisualBrush(surface);
                context.DrawRectangle(brush,
                                      null,
                                      new Rect(new Point(), new Size(surface.Width, surface.Height)));
            }

            visual.Transform = new ScaleTransform(28 / surface.ActualWidth, 28 / surface.ActualHeight);
            renderBitmap.Render(visual);
            var bitmapGreyscale = new FormatConvertedBitmap(renderBitmap, PixelFormats.Gray8, BitmapPalettes.Gray256, 0.0);
            surface.LayoutTransform = transform;
            return bitmapGreyscale;
        }

        private void BrushSizeSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (PaintCanvas == null) return;
            var drawingAttributes = PaintCanvas.DefaultDrawingAttributes;
            drawingAttributes.Width = BrushSlider.Value;
            drawingAttributes.Height = BrushSlider.Value;
            PaintCanvas.EraserShape = new RectangleStylusShape(BrushSlider.Value, BrushSlider.Value);
            var previousEditingMode = PaintCanvas.EditingMode;
            PaintCanvas.EditingMode = InkCanvasEditingMode.None;
            PaintCanvas.EditingMode = previousEditingMode;
        }

        private void BrushColorCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!(BrushColorCombo.SelectedItem is PropertyInfo info)) return;
            var selectedColor = (Color) info.GetValue(null, null);
            PaintCanvas.DefaultDrawingAttributes.Color = selectedColor;
        }

        private void BrushStateCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (PaintCanvas == null) return;
            PaintCanvas.EditingMode = BrushStateCombo.SelectedIndex switch
            {
                0 => InkCanvasEditingMode.Ink,
                1 => InkCanvasEditingMode.Select,
                2 => InkCanvasEditingMode.EraseByPoint,
                3 => InkCanvasEditingMode.EraseByStroke,
                _ => PaintCanvas.EditingMode
            };
        }

        private void BrushShapesCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (PaintCanvas == null) return;
            PaintCanvas.DefaultDrawingAttributes.StylusTip = BrushShapesCombo.SelectedIndex switch
            {
                0 => StylusTip.Ellipse,
                1 => StylusTip.Rectangle,
                _ => PaintCanvas.DefaultDrawingAttributes.StylusTip
            };
        }

        private void MakePredictButton_Click(object sender, RoutedEventArgs e)
        {
            var byteArray = GetImageLikeByteArray(PaintCanvas); //For predict
            var pred = Mnist.Program.Predict(byteArray);
            ResultOfPredict.Text = Mnist.Program.PredictedIndex(pred).ToString();//network output
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            SaveToFilePng(new Uri(Directory.GetCurrentDirectory() + "/file1.png"), PaintCanvas);
        }

        private void TrainButton_Click(object sender, RoutedEventArgs e)
        {

        }

        private void ChooseFolderButton_Click(object sender, RoutedEventArgs e)
        {
            Mnist.Program.modelPath = TextModelFolder.Text;
        }
    }
}