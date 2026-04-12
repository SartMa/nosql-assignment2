import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

import javax.naming.Context;

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

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);
        private final Text word = new Text();
        private final Set<String> stopWords = new HashSet<String>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            if (!conf.getBoolean("wordcount.skip.patterns", false)) {
                return;
            }

            URI[] patternsURIs = context.getCacheFiles();
            if (patternsURIs == null || patternsURIs.length == 0) {
                throw new IOException("Stopwords cache file is missing. Add it via job.addCacheFile(...)");
            }

            for (URI patternsURI : patternsURIs) {
                Path patternsPath = new Path(patternsURI.getPath());
                String patternsFileName = patternsPath.getName().toString();
                parseStopWordsFile(patternsFileName);
            }
        }

        private void parseStopWordsFile(String fileName) throws IOException {
            try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
                String pattern;
                while ((pattern = reader.readLine()) != null) {
                    if (pattern.trim().length() > 0) {
                        stopWords.add(pattern.trim().toLowerCase());
                    }
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^\\w']+");
            for (String token : tokens) {
                if (token.length() > 0 && !stopWords.contains(token)) {
                    word.set(token);
                    context.write(word, ONE);
                }
            }
        }
    }

    public static class SumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable outValue = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            outValue.set(sum);
            context.write(key, outValue);
        }
    }

    public static class Top50Reducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private static final int TOP_K = 50;
        private final PriorityQueue<WordFreq> topWords = new PriorityQueue<WordFreq>();
        private final Text outKey = new Text();
        private final IntWritable outValue = new IntWritable();

        private static class WordFreq implements Comparable<WordFreq> {
            final String word;
            final int freq;

            WordFreq(String word, int freq) {
                this.word = word;
                this.freq = freq;
            }

            @Override
            public int compareTo(WordFreq other) {
                int byFreq = Integer.compare(this.freq, other.freq);
                if (byFreq != 0) {
                    return byFreq;
                }
                return other.word.compareTo(this.word);
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
            if (topWords.size() > TOP_K) {
                topWords.poll();
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            List<WordFreq> words = new ArrayList<WordFreq>(topWords);
            words.sort((a, b) -> {
                int byFreq = Integer.compare(b.freq, a.freq);
                if (byFreq != 0) {
                    return byFreq;
                }
                return a.word.compareTo(b.word);
            });

            for (WordFreq wf : words) {
                outKey.set(wf.word);
                outValue.set(wf.freq);
                context.write(outKey, outValue);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: Problem1A <in> <out> <stopwords_path>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.setBoolean("wordcount.skip.patterns", true);

        Job job = Job.getInstance(conf, "problem1a-top50-frequent-words");
        job.setJarByClass(Problem1A.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(SumCombiner.class);
        job.setReducerClass(Top50Reducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.addCacheFile(new Path(args[2]).toUri());

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}