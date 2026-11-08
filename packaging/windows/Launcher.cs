using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: AssemblyTitle("Image Analysis")]
[assembly: AssemblyVersion("0.3.8.0")]

internal static class Program
{
    internal static readonly string Home = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ImageAnalysis");
    internal static readonly string AppRoot = Path.Combine(Home, "app");
    internal static readonly string OfflineRoot = Path.Combine(
        Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "offline");
#if OFFLINE
    internal static readonly bool IsOffline = true;
#else
    internal static readonly bool IsOffline = Directory.Exists(OfflineRoot);
#endif

    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        bool owns;
        using (var mutex = new Mutex(true, @"Local\ImageAnalysisLauncher", out owns))
        {
            if (!owns) { MessageBox.Show("Окно Image Analysis уже открыто."); return 1; }
            try
            {
                if (args.Length == 2 && args[0] == "--self-test")
                {
                    // Isolated test destination, never the installed application.
                    string root = Path.GetFullPath(args[1]);
                    Extract(root);
                    using (var window = new Launcher())
                    {
                        window.Show();
                        Application.DoEvents();
                        using (var bitmap = new Bitmap(window.Width, window.Height))
                        {
                            window.DrawToBitmap(bitmap, new Rectangle(Point.Empty, window.Size));
                            bitmap.Save(Path.Combine(root, "launcher.png"));
                        }
                        window.Close();
                    }
                    var info = Python(root, "--action self-test");
                    using (var process = Process.Start(info))
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string errors = process.StandardError.ReadToEnd();
                        process.WaitForExit();
                        File.WriteAllText(Path.Combine(root, "self-test.txt"), "exit=" + process.ExitCode + "\n" + output + errors);
                        return process.ExitCode;
                    }
                }
                Application.Run(new Launcher());
                return 0;
            }
            catch (Exception ex)
            {
                if (args.Length == 2 && args[0] == "--self-test")
                {
                    Directory.CreateDirectory(args[1]);
                    File.WriteAllText(Path.Combine(args[1], "self-test.txt"), ex.ToString());
                }
                else MessageBox.Show("Не удалось открыть мастер настройки Image Analysis.", "Ошибка Image Analysis", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }
    }

    internal static void Extract(string root)
    {
        Directory.CreateDirectory(root);
        using (var payload = Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip"))
        {
            string hash;
            using (var sha = SHA256.Create()) { hash = BitConverter.ToString(sha.ComputeHash(payload)); }
            string marker = Path.Combine(root, ".payload-sha256");
            if (File.Exists(marker) && File.ReadAllText(marker) == hash) return;
            payload.Position = 0;
            using (var zip = new ZipArchive(payload, ZipArchiveMode.Read))
            {
                string prefix = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
                foreach (var entry in zip.Entries)
                {
                    string destination = Path.GetFullPath(Path.Combine(root, entry.FullName));
                    if (!destination.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException("Недопустимый путь в поставке.");
                    Directory.CreateDirectory(Path.GetDirectoryName(destination));
                    using (var source = entry.Open())
                    using (var output = File.Create(destination)) { source.CopyTo(output); }
                }
            }
            // Only a complete extraction can be reused on the next launch.
            File.WriteAllText(marker, hash);
        }
    }

    internal static ProcessStartInfo Python(string root, string arguments)
    {
        var info = new ProcessStartInfo(Path.Combine(root, "python", "python.exe"),
            "-u scripts/windows_setup.py " + arguments);
        info.WorkingDirectory = root;
        info.UseShellExecute = false;
        info.CreateNoWindow = true;
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;
        info.StandardOutputEncoding = Encoding.UTF8;
        info.StandardErrorEncoding = Encoding.UTF8;
        info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        info.EnvironmentVariables["PYTHONUTF8"] = "1";
        info.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
        if (IsOffline) info.EnvironmentVariables["IMAGE_ANALYSIS_OFFLINE_DIR"] = OfflineRoot;
        return info;
    }
}

internal sealed class Launcher : Form
{
    private readonly Label message = new Label();
    private readonly ComboBox device = new ComboBox();
    private readonly ProgressBar progress = new ProgressBar();
    private readonly Button start = new Button();
    private readonly Button stop = new Button();
    private readonly Button wsl = new Button();
    private readonly Button journal = new Button();
    private bool busy;
    private string latestLog;

    internal Launcher()
    {
        Text = "Image Analysis 0.3.8";
        Font = new Font("Segoe UI", 11);
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(740, 455);
        MinimumSize = Size;
        StartPosition = FormStartPosition.CenterScreen;
        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(24), ColumnCount = 1, RowCount = 7 };
        layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 106));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 56));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 64));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
        Controls.Add(layout);
        layout.Controls.Add(new Label { Text = "Анализ фотографий на вашем компьютере", Dock = DockStyle.Fill,
            Font = new Font(Font, FontStyle.Bold) });
        layout.Controls.Add(new Label { Text = Program.IsOffline
            ? "Нажмите «Запустить»: мастер установит Docker, модели и зависимости из этой поставки и откроет браузер.\n\nИнтернет не требуется. Рекомендуется 16 ГБ RAM и 50 ГБ свободного диска с учётом распаковки. Возможна перезагрузка Windows."
            : "Нажмите «Запустить»: мастер подготовит Docker, скачает модели и откроет приложение в браузере.\n\nПервая установка требует интернета, около 30 ГБ свободного диска и рекомендуется 16 ГБ RAM. Возможна перезагрузка Windows.",
            Dock = DockStyle.Fill });
        var selection = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
        selection.Controls.Add(new Label { Text = "Обработка:", AutoSize = true, Margin = new Padding(0, 7, 12, 0) });
        device.DropDownStyle = ComboBoxStyle.DropDownList;
        device.Width = 460;
        device.AccessibleName = "Устройство обработки";
        device.Items.AddRange(new object[] { "CPU — без отдельной видеокарты, медленнее", "NVIDIA GPU — нужен совместимый драйвер и Docker GPU" });
        string saved = Path.Combine(Program.Home, "device.txt");
        device.SelectedIndex = File.Exists(saved) && File.ReadAllText(saved).Trim() == "cuda" ? 1 : 0;
        selection.Controls.Add(device);
        layout.Controls.Add(selection);
        message.Text = "Готово к настройке. Хранилище на этом компьютере создаётся автоматически.";
        message.Dock = DockStyle.Fill;
        message.AccessibleName = "Состояние настройки";
        layout.Controls.Add(message);
        progress.Dock = DockStyle.Fill;
        progress.AccessibleName = "Выполнение настройки";
        layout.Controls.Add(progress);
        var buttons = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false, Padding = new Padding(0, 12, 0, 0) };
        start.Text = "&Запустить";
        stop.Text = "&Остановить";
        wsl.Text = "Настроить &WSL";
        journal.Text = "&Журнал";
        foreach (var button in new[] { start, stop, wsl, journal })
        {
            button.Size = new Size(154, 40);
            button.Margin = new Padding(0, 0, 12, 0);
            buttons.Controls.Add(button);
        }
        layout.Controls.Add(buttons);
        layout.Controls.Add(new Label { Text = "Закрытие этого окна после запуска оставляет приложение работающим.", Dock = DockStyle.Fill });
        AcceptButton = start;
        start.Click += async (sender, args) => await Run("start");
        stop.Click += async (sender, args) => await Run("stop");
        wsl.Click += async (sender, args) => await Run("wsl");
        journal.Click += (sender, args) =>
        {
            if (latestLog != null) Process.Start("notepad.exe", "\"" + latestLog + "\"");
            else MessageBox.Show(this, "Журнал появится после запуска настройки.", "Image Analysis");
        };
        FormClosing += (sender, args) =>
        {
            if (!busy || args.CloseReason == CloseReason.WindowsShutDown) return;
            args.Cancel = true;
            MessageBox.Show(this, "Дождитесь завершения текущего этапа настройки.", "Настройка выполняется");
        };
    }

    private void UpdateMessage(string text)
    {
        BeginInvoke((Action)(() => message.Text = text));
    }

    private async Task Run(string action)
    {
        if (busy) return;
        busy = true;
        start.Enabled = stop.Enabled = wsl.Enabled = device.Enabled = false;
        progress.Style = ProgressBarStyle.Marquee;
        message.Text = "Подготовка файлов приложения…";
        string mode = device.SelectedIndex == 1 ? "cuda" : "cpu";
        string error = null;
        try
        {
            Directory.CreateDirectory(Path.Combine(Program.Home, "logs"));
            latestLog = Path.Combine(Program.Home, "logs", DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + ".log");
            string logPath = latestLog;
            int exitCode = await Task.Run(() =>
            {
                Program.Extract(Program.AppRoot);
                File.WriteAllText(Path.Combine(Program.Home, "device.txt"), mode);
                using (var log = new StreamWriter(logPath, false, new UTF8Encoding(true)))
                {
                    log.AutoFlush = true;
                    using (var process = new Process { StartInfo = Program.Python(Program.AppRoot,
                        "--action " + action + " --device " + mode) })
                    {
                        DataReceivedEventHandler receive = (sender, args) =>
                        {
                            if (args.Data == null) return;
                            lock (log) { log.WriteLine(args.Data); }
                            if (args.Data.StartsWith("@@STATUS ")) UpdateMessage(args.Data.Substring(9));
                            if (args.Data.StartsWith("@@ERROR ")) error = args.Data.Substring(8);
                        };
                        process.OutputDataReceived += receive;
                        process.ErrorDataReceived += receive;
                        process.Start();
                        process.BeginOutputReadLine();
                        process.BeginErrorReadLine();
                        process.WaitForExit();
                        return process.ExitCode;
                    }
                }
            });
            if (exitCode != 0) message.Text = error ?? "Настройка прервана. Откройте журнал, устраните причину и повторите запуск.";
        }
        catch (Exception ex)
        {
            message.Text = "Не удалось подготовить файлы приложения. Проверьте свободное место и откройте журнал.";
            try { if (latestLog != null) File.AppendAllText(latestLog, ex.ToString()); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
        finally
        {
            busy = false;
            start.Enabled = stop.Enabled = wsl.Enabled = device.Enabled = true;
            progress.Style = ProgressBarStyle.Blocks;
            start.Focus();
        }
    }
}
