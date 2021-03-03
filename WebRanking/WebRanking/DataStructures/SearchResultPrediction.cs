using System;
using System.Collections.Generic;
using System.Text;

namespace WebRanking.DataStructures
{
    class SearchResultPrediction
    {
        public uint GroupId { get; set; }

        public uint Label { get; set; }

        public float Score { get; set; }

        public float[] Features
        {
            get; set;
        }
    }
}
