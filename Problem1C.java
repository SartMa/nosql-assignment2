import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Problem1C {

    // =========================
    // MAPPER
    // =========================
    public static class StripeMapper extends Mapper<Object, Text, Text, MapWritable> {

        private Set<String> topWords = new HashSet<>();
        private int window;

        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            window = conf.getInt("window", 1);

            URI[] files = context.getCacheFiles();
            if (files == null || files.length == 0) {
                throw new IOException("Top-50 cache file is missing. Add it via job.addCacheFile(...)");
            }

            File localizedFile = new File(new Path(files[0].getPath()).getName());
            if (!localizedFile.exists()) {
                localizedFile = new File(files[0].getPath());
            }

            if (localizedFile.exists()) {
                try (BufferedReader br = new BufferedReader(new FileReader(localizedFile))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] parts = line.trim().split("\\s+");
                        if (parts.length == 0 || parts[0].isEmpty()) {
                            continue;
                        }
                        topWords.add(parts[0].toLowerCase());
                    }
                }
            } else {
                // Fallback for local mode runners where cache files may not be localized.
                Path cachePath = new Path(files[0]);
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(cachePath.getFileSystem(conf).open(cachePath)))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] parts = line.trim().split("\\s+");
                        if (parts.length == 0 || parts[0].isEmpty()) {
                            continue;
                        }
                        topWords.add(parts[0].toLowerCase());
                    }
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            String[] tokens = value.toString().toLowerCase().split("[^\\w']+");

            for (int i = 0; i < tokens.length; i++) {

                String wi = tokens[i];
                if (wi.length() == 0 || !topWords.contains(wi)) continue;

                MapWritable stripe = new MapWritable();

                int start = Math.max(0, i - window);
                int end = Math.min(tokens.length - 1, i + window);

                for (int j = start; j <= end; j++) {
                    if (i == j) continue;

                    String wj = tokens[j];
                    if (wj.length() == 0 || !topWords.contains(wj)) continue;

                    Text neighbor = new Text(wj);

                    if (stripe.containsKey(neighbor)) {
                        IntWritable count = (IntWritable) stripe.get(neighbor);
                        stripe.put(neighbor, new IntWritable(count.get() + 1));
                    } else {
                        stripe.put(neighbor, new IntWritable(1));
                    }
                }

                context.write(new Text(wi), stripe);
            }
        }
    }

    // =========================
    // REDUCER
    // =========================
    public static class StripeReducer extends Reducer<Text, MapWritable, Text, MapWritable> {

        @Override
        public void reduce(Text key, Iterable<MapWritable> values, Context context)
                throws IOException, InterruptedException {

            MapWritable result = new MapWritable();

            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {

                    Text neighbor = (Text) entry.getKey();
                    IntWritable count = (IntWritable) entry.getValue();

                    if (result.containsKey(neighbor)) {
                        IntWritable existing = (IntWritable) result.get(neighbor);
                        result.put(neighbor, new IntWritable(existing.get() + count.get()));
                    } else {
                        result.put(neighbor, new IntWritable(count.get()));
                    }
                }
            }

            context.write(key, result);
        }
    }

    // =========================
    // MAIN
    // =========================
    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            System.err.println("Usage: Problem1C <input> <output> <top50file> <window>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.setInt("window", Integer.parseInt(args[3]));

        long startTime = System.currentTimeMillis();

        Job job = Job.getInstance(conf, "Stripes Co-occurrence");

        job.setJarByClass(Problem1C.class);

        job.setMapperClass(StripeMapper.class);
        job.setReducerClass(StripeReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MapWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(MapWritable.class);

        job.addCacheFile(new Path(args[2]).toUri());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean success = job.waitForCompletion(true);

        long endTime = System.currentTimeMillis();
        double runtime = (endTime - startTime) / 1000.0;

        System.err.println("=== TOTAL RUNTIME (sec): " + runtime + " ===");

        System.exit(success ? 0 : 1);
    }
}