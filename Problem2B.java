import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import opennlp.tools.stemmer.PorterStemmer;

public class Problem2B {

    private static final double DOC_NORMALIZER = 10000.0;

    private static void loadDfFromCache(URI[] cacheFiles, Map<String, Integer> dfByTerm) throws IOException {
        if (cacheFiles == null || cacheFiles.length == 0) {
            throw new IOException("DF cache file is missing. Pass top100_df.tsv as the third argument.");
        }

        for (URI cacheFile : cacheFiles) {
            Path cachePath = new Path(cacheFile.getPath());
            File localFile = new File(cachePath.getName());
            if (!localFile.exists()) {
                localFile = new File(cacheFile.getPath());
            }

            if (!localFile.exists()) {
                continue;
            }

            try (BufferedReader reader = new BufferedReader(new FileReader(localFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split("\\t");
                    if (parts.length < 2) {
                        continue;
                    }

                    String term = parts[0].trim();
                    if (term.isEmpty()) {
                        continue;
                    }

                    try {
                        int df = Integer.parseInt(parts[1].trim());
                        if (df > 0) {
                            dfByTerm.put(term, df);
                        }
                    } catch (NumberFormatException ignored) {
                        // Skip malformed lines.
                    }
                }
            }
        }

        if (dfByTerm.isEmpty()) {
            throw new IOException("No DF entries were loaded from cached TSV file.");
        }
    }

    public static class TermFrequencyMapper extends Mapper<LongWritable, Text, Text, MapWritable> {

        private final Text docIdText = new Text();
        private final Map<String, Integer> dfByTerm = new HashMap<>();
        private PorterStemmer stemmer;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            stemmer = new PorterStemmer();
            loadDfFromCache(context.getCacheFiles(), dfByTerm);

            FileSplit split = (FileSplit) context.getInputSplit();
            docIdText.set(split.getPath().getName());
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase();
            String[] tokens = line.split("[^a-z]+");

            Map<String, Integer> localStripe = new HashMap<>();
            for (String token : tokens) {
                if (token.isEmpty()) {
                    continue;
                }

                String stemmed = stemmer.stem(token);
                if (!dfByTerm.containsKey(stemmed)) {
                    continue;
                }

                int current = localStripe.containsKey(stemmed) ? localStripe.get(stemmed) : 0;
                localStripe.put(stemmed, current + 1);
            }

            if (localStripe.isEmpty()) {
                return;
            }

            MapWritable stripeWritable = new MapWritable();
            for (Map.Entry<String, Integer> entry : localStripe.entrySet()) {
                stripeWritable.put(new Text(entry.getKey()), new IntWritable(entry.getValue()));
            }

            context.write(docIdText, stripeWritable);
        }
    }

    public static class StripeSumCombiner extends Reducer<Text, MapWritable, Text, MapWritable> {
        @Override
        protected void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {
            Map<String, Integer> summedStripe = new HashMap<>();

            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    String term = entry.getKey().toString();
                    int tf = ((IntWritable) entry.getValue()).get();
                    int current = summedStripe.containsKey(term) ? summedStripe.get(term) : 0;
                    summedStripe.put(term, current + tf);
                }
            }

            if (summedStripe.isEmpty()) {
                return;
            }

            MapWritable resultStripe = new MapWritable();
            for (Map.Entry<String, Integer> entry : summedStripe.entrySet()) {
                resultStripe.put(new Text(entry.getKey()), new IntWritable(entry.getValue()));
            }

            context.write(key, resultStripe);
        }
    }

    public static class TfDfScoreReducer extends Reducer<Text, MapWritable, Text, DoubleWritable> {

        private final Text outputKey = new Text();
        private final DoubleWritable outputScore = new DoubleWritable();
        private final Map<String, Integer> dfByTerm = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            loadDfFromCache(context.getCacheFiles(), dfByTerm);
        }

        @Override
        protected void reduce(Text docId, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            Map<String, Integer> tfByTerm = new HashMap<>();
            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    String term = entry.getKey().toString();
                    int tf = ((IntWritable) entry.getValue()).get();
                    int current = tfByTerm.containsKey(term) ? tfByTerm.get(term) : 0;
                    tfByTerm.put(term, current + tf);
                }
            }

            String docIdStr = docId.toString();
            for (Map.Entry<String, Integer> tfEntry : tfByTerm.entrySet()) {
                String term = tfEntry.getKey();
                int tf = tfEntry.getValue();
                Integer df = dfByTerm.get(term);
                if (df == null || df <= 0) {
                    continue;
                }

                double score = tf * Math.log((DOC_NORMALIZER / df) + 1.0);

                outputKey.set(docIdStr + "\t" + term);
                outputScore.set(score);
                context.write(outputKey, outputScore);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: Problem2B <input path> <output path> <top100_df.tsv path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        conf.set("mapreduce.output.textoutputformat.separator", "\t");

        Job job = Job.getInstance(conf, "Problem2B TF-DF scoring");
        job.setJarByClass(Problem2B.class);

        job.setMapperClass(TermFrequencyMapper.class);
        job.setCombinerClass(StripeSumCombiner.class);
        job.setReducerClass(TfDfScoreReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        job.addCacheFile(new Path(args[2]).toUri());

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}