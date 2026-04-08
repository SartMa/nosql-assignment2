import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Problem1A {

    private static final String DEFAULT_LOG_FILE = "problem1a.log";
    private static final DateTimeFormatter LOG_TIMESTAMP_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static synchronized void logMessage(Configuration conf, String message) {
        String timestampedMessage = "[" + LOG_TIMESTAMP_FORMAT.format(LocalDateTime.now()) + "] " + message;
        System.err.println(timestampedMessage);

        String logFilePath = DEFAULT_LOG_FILE;
        if (conf != null) {
            logFilePath = conf.get("problem1a.log.path", DEFAULT_LOG_FILE);
        }

        try {
            Files.write(
                    Paths.get(logFilePath),
                    (timestampedMessage + System.lineSeparator()).getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException ioe) {
            System.err.println("Failed to write log file '" + logFilePath + "': " + ioe.getMessage());
        }
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private Set<String> stopWords = new HashSet<String>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            URI[] patternsURIs = context.getCacheFiles();
            if (patternsURIs == null || patternsURIs.length == 0) {
                Problem1A.logMessage(context.getConfiguration(), "No stopword file found in distributed cache.");
                return;
            }

            for (URI patternsURI : patternsURIs) {
                Path patternsPath = new Path(patternsURI.getPath());
                String patternsFileName = patternsPath.getName().toString();
                parseStopWordsFile(patternsFileName, context);
            }

            Problem1A.logMessage(context.getConfiguration(), "Loaded " + stopWords.size() + " stopwords.");
        }

        private void parseStopWordsFile(String fileName, Context context) throws IOException {
            try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
                String pattern;
                while ((pattern = reader.readLine()) != null) {
                    if (pattern.trim().length() > 0) {
                        stopWords.add(pattern.trim().toLowerCase());
                    }
                }
            } catch (IOException ioe) {
                Problem1A.logMessage(
                        context.getConfiguration(),
                        "Caught exception while parsing cached file '" + fileName + "': " + ioe.getMessage());
                throw ioe;
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-zA-Z]+");
            for (String token : tokens) {
                if (token.length() > 0 && !stopWords.contains(token)) {
                    word.set(token);
                    context.write(word, one);
                }
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private PriorityQueue<WordFreq> topWords = new PriorityQueue<>();

        static class WordFreq implements Comparable<WordFreq> {
            String word;
            int freq;

            WordFreq(String word, int freq) {
                this.word = word;
                this.freq = freq;
            }

            @Override
            public int compareTo(WordFreq other) {
                return Integer.compare(this.freq, other.freq);
            }
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            
            topWords.add(new WordFreq(key.toString(), sum));
            if (topWords.size() > 50) {
                topWords.poll();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            WordFreq[] words = topWords.toArray(new WordFreq[0]);
            java.util.Arrays.sort(words, (a, b) -> Integer.compare(b.freq, a.freq));
            for (WordFreq wf : words) {
                context.write(new Text(wf.word), new IntWritable(wf.freq));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3 || args.length > 4) {
            System.err.println("Usage: Problem1A <in> <out> <stopwords_path> [log_file]");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        if (args.length == 4) {
            conf.set("problem1a.log.path", args[3]);
        }

        logMessage(conf, "Starting Problem1A job.");

        Job job = Job.getInstance(conf, "top 50 word count");
        job.setJarByClass(Problem1A.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.addCacheFile(new Path(args[2]).toUri());

        logMessage(conf, "Input=" + args[0] + ", Output=" + args[1] + ", Stopwords=" + args[2]);

        boolean success = job.waitForCompletion(true);
        logMessage(conf, "Problem1A job finished with status: " + (success ? "SUCCESS" : "FAILED"));

        System.exit(success ? 0 : 1);
    }
}